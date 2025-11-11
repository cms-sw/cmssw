/*
 * CandidateSimMuonMatcher.cc
 *
 *  Created on: Dec 14, 2020
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/CandidateSimMuonMatcher.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "boost/dynamic_bitset.hpp"

#include "TFile.h"
#include "TH1D.h"
/*
double hwGmtPhiToGlobalPhi(int phi) {
  double phiGmtUnit = 2. * M_PI / 576.;
  double globalPhi = phi * phiGmtUnit;

  if (globalPhi > M_PI)
    globalPhi = globalPhi - (2. * M_PI);

  return globalPhi;
}*/

double foldPhi(double phi) {
  if (phi > M_PI)
    return (phi - 2 * M_PI);
  else if (phi < -M_PI)
    return (phi + 2 * M_PI);

  return phi;
}

MatchingResult::MatchingResult(const SimTrack& simTrack, const SimVertex* simVertex)
    : simTrack(&simTrack), simVertex(simVertex) {
  pdgId = simTrack.type();
  genPt = simTrack.momentum().pt();
  genEta = simTrack.momentum().eta();
  genPhi = simTrack.momentum().phi();
  genCharge = simTrack.charge();

  if (this->simVertex) {
    const math::XYZTLorentzVectorD& vtxPos = this->simVertex->position();
    muonDxy = (-vtxPos.X() * this->simTrack->momentum().py() + vtxPos.Y() * this->simTrack->momentum().px()) /
              this->simTrack->momentum().pt();
    muonRho = vtxPos.Rho();

    vertexEta = vtxPos.eta();
    vertexPhi = vtxPos.phi();
  } else {
    // default values when no vertex information is available
    muonDxy = 0;
    muonRho = 0;
    vertexEta = 0;
    vertexPhi = 0;
  }

  //parentPdgId = 0; TODO
}

MatchingResult::MatchingResult(const TrackingParticle& trackingParticle) : trackingParticle(&trackingParticle) {
  pdgId = trackingParticle.pdgId();
  genPt = trackingParticle.pt();
  genEta = trackingParticle.momentum().eta();
  genPhi = trackingParticle.momentum().phi();
  genCharge = trackingParticle.charge();

  muonDxy = this->trackingParticle->dxy();
  if (this->trackingParticle->parentVertex().isNonnull()) {
    muonRho = this->trackingParticle->parentVertex()->position().Rho();
    vertexEta = this->trackingParticle->parentVertex()->position().eta();
    vertexPhi = this->trackingParticle->parentVertex()->position().phi();
  } else {
    muonRho = 0;
    vertexEta = 0;
    vertexPhi = 0;
  }
}

std::ostream& operator<<(std::ostream& out, const MatchingResult& matchingResult) {
  out << "matchingResult:\n";
  if (matchingResult.simTrack || matchingResult.trackingParticle) {
    out << " simTrack type " << std::setw(5) << matchingResult.pdgId;
    out << " pt " << std::setw(8) << matchingResult.genPt << " GeV";
    out << " eta " << std::setw(8) << matchingResult.genEta;
    out << " phi " << std::setw(8) << matchingResult.genPhi;
    out << " charge " << std::setw(2) << matchingResult.genCharge;
    out << " vertexEta " << std::setw(8) << matchingResult.vertexEta;
    out << " vertexPhi " << std::setw(8) << matchingResult.vertexPhi;
    out << " muonDxy " << std::setw(8) << matchingResult.muonDxy;
    out << " muonRho " << std::setw(8) << matchingResult.muonRho;
    out << " parentPdgId " << std::setw(5) << matchingResult.parentPdgId;
  } else {
    out << " no matched simTrack/trackingParticle";
  }
  out << "\n muonCand: ";
  if (matchingResult.muonCand)
    out << *(matchingResult.muonCand);
  else
    out << " no matched muonCand ";

  out << "\n deltaEta " << std::setw(8) << matchingResult.deltaEta << " deltaPhi " << std::setw(8)
      << matchingResult.deltaPhi << " Likelihood " << std::setw(8) << matchingResult.matchingLikelihood << " result "
      << (short)matchingResult.result << std::endl;
  return out;
}

int CandidateSimMuonMatcher::calcGlobalPhi(int locPhi, int proc) {
  int globPhi = 0;
  //60 degree sectors = 96 in int-scale
  globPhi = (proc) * 96 * 6 / nProcessors + locPhi;
  // first processor starts at CMS phi = 15 degrees (24 in int)... Handle wrap-around with %. Add 576 to make sure the number is positive
  globPhi = (globPhi + 24 + 576) % 576;
  return globPhi;
}

CandidateSimMuonMatcher::CandidateSimMuonMatcher(
    const edm::ParameterSet& edmCfg,
    //const OMTFConfiguration* omtfConfig,
    int nProcessors,
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
    const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken)
    :  //omtfConfig(omtfConfig),
      nProcessors(nProcessors),
      edmCfg(edmCfg),
      magneticFieldEsToken(magneticFieldEsToken),
      propagatorEsToken(propagatorEsToken) {
  //needed when CandidateSimMuonMatcher is used outside the OMTF package, where the omtfConfig is not available
  if (edmCfg.exists("phase")) {
    if (edmCfg.getParameter<int>("phase") == 2)
      this->nProcessors = 3;
  }

  std::string muonMatcherFileName = edmCfg.getParameter<edm::FileInPath>("muonMatcherFile").fullPath();
  TFile muonMatcherFile(muonMatcherFileName.c_str());
  edm::LogImportant("l1tOmtfEventPrint") << " CandidateSimMuonMatcher: using muonMatcherFileName "
                                         << muonMatcherFileName << std::endl;
  if (edmCfg.exists("candidateSimMuonMatcherType")) {
    if (edmCfg.getParameter<std::string>("candidateSimMuonMatcherType") == "withPropagator")
      matchingType = MatchingType::withPropagator;
    else if (edmCfg.getParameter<std::string>("candidateSimMuonMatcherType") == "simplePropagation")
      matchingType = MatchingType::simplePropagation;
    else if (edmCfg.getParameter<std::string>("candidateSimMuonMatcherType") == "simpleMatching")
      matchingType = MatchingType::simpleMatching;
    else if (edmCfg.getParameter<std::string>("candidateSimMuonMatcherType") == "collectMuonCands")
      matchingType = MatchingType::collectMuonCands;

    edm::LogImportant("l1tOmtfEventPrint")
        << " CandidateSimMuonMatcher: candidateSimMuonMatcherType "
        << edmCfg.getParameter<std::string>("candidateSimMuonMatcherType") << std::endl;
  }

  edm::LogImportant("l1tOmtfEventPrint") << " CandidateSimMuonMatcher: matchingType "
                                         << static_cast<short>(matchingType) << std::endl;

  // Try to read histograms from the file; be defensive if file/histograms are missing
  if (muonMatcherFile.IsZombie()) {
    edm::LogWarning("l1tOmtfEventPrint") << "CandidateSimMuonMatcher: muon matcher file could not be opened: "
                                         << muonMatcherFileName << std::endl;
    throw cms::Exception("MissingFile") << "CandidateSimMuonMatcher: muon matcher file could not be opened: "
                                        << muonMatcherFileName << std::endl;
  } else {
    edm::LogImportant("l1tOmtfEventPrint") << " CandidateSimMuonMatcher: reading histograms from file " << std::endl;
    minDelta_pos = dynamic_cast<TH1*>(muonMatcherFile.Get("minDelta_pos"));
    LogTrace("l1tOmtfEventPrint") << " CandidateSimMuonMatcher: " << __LINE__ << std::endl;
    maxDelta_pos = dynamic_cast<TH1*>(muonMatcherFile.Get("maxDelta_pos"));
    medianDelta_pos = dynamic_cast<TH1*>(muonMatcherFile.Get("medianDelta_pos"));

    LogTrace("l1tOmtfEventPrint") << " CandidateSimMuonMatcher: " << __LINE__ << std::endl;
    minDelta_neg = dynamic_cast<TH1*>(muonMatcherFile.Get("minDelta_neg"));
    maxDelta_neg = dynamic_cast<TH1*>(muonMatcherFile.Get("maxDelta_neg"));
    medianDelta_neg = dynamic_cast<TH1*>(muonMatcherFile.Get("medianDelta_neg"));

    LogTrace("l1tOmtfEventPrint") << " CandidateSimMuonMatcher: " << __LINE__ << std::endl;
    if (!minDelta_pos || !maxDelta_pos || !medianDelta_pos || !minDelta_neg || !maxDelta_neg || !medianDelta_neg) {
      edm::LogWarning("l1tOmtfEventPrint")
          << "CandidateSimMuonMatcher: one or more histograms are missing in the file: " << muonMatcherFileName
          << std::endl;
      throw cms::Exception("MissingHistogram")
          << "CandidateSimMuonMatcher: one or more histograms are missing in the file: " << muonMatcherFileName
          << std::endl;
    }

    // Detach histograms from the file so they survive file close
    //it must be like that to assure the histograms leave outside this scope.
    //muonMatcherFile as pointer-field in the class somehow does not work.
    minDelta_pos->SetDirectory(nullptr);
    maxDelta_pos->SetDirectory(nullptr);
    medianDelta_pos->SetDirectory(nullptr);
    minDelta_neg->SetDirectory(nullptr);
    maxDelta_neg->SetDirectory(nullptr);
    medianDelta_neg->SetDirectory(nullptr);
  }
}

CandidateSimMuonMatcher::~CandidateSimMuonMatcher() {
  if (minDelta_pos)
    delete minDelta_pos;
  if (maxDelta_pos)
    delete maxDelta_pos;
  if (medianDelta_pos)
    delete medianDelta_pos;
  if (minDelta_neg)
    delete minDelta_neg;
  if (maxDelta_neg)
    delete maxDelta_neg;
  if (medianDelta_neg)
    delete medianDelta_neg;
}

void CandidateSimMuonMatcher::beginRun(const edm::EventSetup& eventSetup) {
  //TODO use edm::ESWatcher<MagneticField> magneticFieldRecordWatcher;
  magField = eventSetup.getHandle(magneticFieldEsToken);
  propagator = eventSetup.getHandle(propagatorEsToken);
}

void CandidateSimMuonMatcher::observeEventBegin(const edm::Event& event) { gbCandidates.clear(); }

void CandidateSimMuonMatcher::observeProcesorEmulation(unsigned int iProcessor,
                                                       l1t::tftype mtfType,
                                                       const std::shared_ptr<OMTFinput>& input,
                                                       const AlgoMuons& algoCandidates,
                                                       const AlgoMuons& gbCandidates,
                                                       const FinalMuons& finalMuons) {
  //debug, gbCandidate are not used outside this method
  //unsigned int procIndx = omtfConfig->getProcIndx(iProcessor, mtfType);
  for (auto& gbCandidate : gbCandidates) {
    //if (gbCandidate->getPtConstr() > 0)
    {
      LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::observeProcesorEmulation iProcessor " << iProcessor
                                    << " mtfType " << mtfType << *gbCandidate << std::endl;
      this->gbCandidates.emplace_back(gbCandidate);
    }
  }
}

bool simTrackIsMuonInOmtf(const SimTrack& simTrack) {
  if (std::abs(simTrack.type()) == 13 || std::abs(simTrack.type()) == 1000015) {  // 1000015 is stau
    //only muons
  } else
    return false;

  //in the overlap, the propagation of muons with pt less then ~3.2 fails - the actual threshold depends slightly on eta,
  if (simTrack.momentum().pt() < 2.5)
    return false;

  LogTrace("l1tOmtfEventPrint") << "simTrackIsMuonInOmtf: simTrack type " << std::setw(3) << simTrack.type() << " pt "
                                << std::setw(9) << simTrack.momentum().pt() << " eta " << std::setw(9)
                                << simTrack.momentum().eta() << " phi " << std::setw(9) << simTrack.momentum().phi()
                                << std::endl;

  //some margin for matching must be used on top of actual OMTF region,
  //i.e. (0.82-1.24)=>(0.72-1.3),
  //otherwise many candidates are marked as ghosts
  if ((std::abs(simTrack.momentum().eta()) >= 0.7) && (std::abs(simTrack.momentum().eta()) <= 1.31)) {
    LogTrace("l1tOmtfEventPrint") << "simTrackIsMuonInOmtf: is in OMTF";
  } else {
    LogTrace("l1tOmtfEventPrint") << "simTrackIsMuonInOmtf: not in OMTF";
    return false;
  }

  return true;
}

bool simTrackIsMuonInOmtfBx0(const SimTrack& simTrack) {
  if (simTrack.eventId().bunchCrossing() != 0)
    return false;

  return simTrackIsMuonInOmtf(simTrack);
}

bool simTrackIsMuonInBx0(const SimTrack& simTrack) {
  if (std::abs(simTrack.type()) == 13 || std::abs(simTrack.type()) == 1000015) {  // 1000015 is stau
    if (simTrack.momentum().pt() < 2.5)  //TODO muons with pt < 2.5 GeV do not reach the OMTF
      return false;

    //only muons
    if (simTrack.eventId().bunchCrossing() == 0)
      return true;
  }
  return false;
}

bool trackingParticleIsMuonInBx0(const TrackingParticle& trackingParticle) {
  if (std::abs(trackingParticle.pdgId()) == 13 ||
      std::abs(trackingParticle.pdgId()) ==
          1000015) {  // 1000015 is stau, todo use other selection (e.g. pt>20) if needed

    //in the overlap, the propagation of muons with pt less then ~3.2 fails - the actual threshold depends slightly on eta,
    if (trackingParticle.pt() < 2.5)
      return false;

    //only muons
    if (trackingParticle.eventId().bunchCrossing() == 0)
      return true;
  }
  return false;
}

bool trackingParticleIsMuonInBx0Displ(const TrackingParticle& trackingParticle) {
  if (std::abs(trackingParticle.pdgId()) == 13 || std::abs(trackingParticle.pdgId()) == 1000015) {  // 1000015 is stau

    //in the overlap, the propagation of muons with pt less then ~3.2 fails - the actual threshold depends slightly on eta,
    if (trackingParticle.pt() < 2.)
      return false;

    if (trackingParticle.parentVertex().isNonnull()) {
      if (trackingParticle.dxy() < 30) {
        if ((std::abs(trackingParticle.momentum().eta()) < 0.7) || (std::abs(trackingParticle.momentum().eta()) > 1.31))
          return false;
      } else {
        if ((std::abs(trackingParticle.parentVertex()->position().eta()) < 0.7) ||
            (std::abs(trackingParticle.parentVertex()->position().eta()) > 1.4))
          return false;
      }
    } else
      throw cms::Exception("CandidateSimMuonMatcher",
                           "trackingParticleIsMuonInBx0Displ: trackingParticle has no parent vertex");

    //only muons
    if (trackingParticle.eventId().bunchCrossing() == 0)
      return true;
  }
  return false;
}

bool trackingParticleIsMuonInOmtfBx0(const TrackingParticle& trackingParticle) {
  if (trackingParticle.eventId().bunchCrossing() != 0)
    return false;

  if (std::abs(trackingParticle.pdgId()) == 13 || std::abs(trackingParticle.pdgId()) == 1000015) {
    // 1000015 is stau, todo use other selection (e.g. pt>20) if needed
    //only muons
  } else
    return false;

  //in the overlap, the propagation of muons with pt less then ~3.2 fails, the actual threshold depends slightly on eta,
  if (trackingParticle.pt() < 2.5)
    return false;

  if (trackingParticle.parentVertex().isNonnull())
    LogTrace("l1tOmtfEventPrint") << "trackingParticleIsMuonInOmtfBx0, pdgId " << std::setw(3)
                                  << trackingParticle.pdgId() << " pt " << std::setw(9) << trackingParticle.pt()
                                  << " eta " << std::setw(9) << trackingParticle.momentum().eta() << " phi "
                                  << std::setw(9) << trackingParticle.momentum().phi() << " event "
                                  << trackingParticle.eventId().event() << " trackId "
                                  << trackingParticle.g4Tracks().at(0).trackId() << " parentVertex Rho "
                                  << trackingParticle.parentVertex()->position().Rho() << " eta "
                                  << trackingParticle.parentVertex()->position().eta() << " phi "
                                  << trackingParticle.parentVertex()->position().phi() << std::endl;
  else
    LogTrace("l1tOmtfEventPrint") << "trackingParticleIsMuonInOmtfBx0, pdgId " << std::setw(3)
                                  << trackingParticle.pdgId() << " pt " << std::setw(9) << trackingParticle.pt()
                                  << " eta " << std::setw(9) << trackingParticle.momentum().eta() << " phi "
                                  << std::setw(9) << trackingParticle.momentum().phi() << " trackId "
                                  << trackingParticle.g4Tracks().at(0).trackId();

  //some margin for matching must be used on top of actual OMTF region,
  //i.e. (0.82-1.24)=>(0.72-1.3),
  //otherwise many candidates are marked as ghosts
  if ((std::abs(trackingParticle.momentum().eta()) >= 0.7) && (std::abs(trackingParticle.momentum().eta()) <= 1.31)) {
  } else
    return false;

  return true;
}

bool trackingParticleIsMuonInOmtfEvent0(const TrackingParticle& trackingParticle) {
  if (trackingParticle.eventId().event() != 0)
    return false;

  return trackingParticleIsMuonInOmtfBx0(trackingParticle);
}

void CandidateSimMuonMatcher::observeEventEnd(const edm::Event& event, FinalMuons& finalMuons) {
  LogTrace("l1tOmtfEventPrint") << "\nCandidateSimMuonMatcher::observeEventEnd" << std::endl;
  FinalMuons ghostBustedFinalMuons = CandidateSimMuonMatcher::ghostBust(finalMuons);

  matchingResults.clear();
  if (edmCfg.exists("simTracksTag")) {
    edm::Handle<edm::SimTrackContainer> simTraksHandle;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("simTracksTag"), simTraksHandle);

    edm::Handle<edm::SimVertexContainer> simVertices;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("simVertexesTag"), simVertices);

    LogTrace("l1tOmtfEventPrint") << "simTraksHandle size " << simTraksHandle.product()->size() << std::endl;
    LogTrace("l1tOmtfEventPrint") << "simVertices size " << simVertices.product()->size() << std::endl;

    if (matchingType == MatchingType::simpleMatching) {
      std::function<bool(const SimTrack&)> const& simTrackFilter = simTrackIsMuonInOmtfBx0;
      //simTrackIsMuonInOmtfBx0 provides appropriate eta cut

      matchingResults =
          matchSimple(ghostBustedFinalMuons, simTraksHandle.product(), simVertices.product(), simTrackFilter);
    } else if (matchingType == MatchingType::withPropagator || matchingType == MatchingType::simplePropagation) {
      //TODO  use other simTrackFilter if needed  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      //withPropagator, we dont want to check the eta of the generated muon, as it is on the vertex,
      //instead inside match, we check the eta of the propagated track to the second muons station
      std::function<bool(const SimTrack&)> const& simTrackFilter =
          (matchingType == MatchingType::withPropagator) ? simTrackIsMuonInBx0 : simTrackIsMuonInOmtfBx0;
      //withPropagator : simplePropagation

      matchingResults = match(ghostBustedFinalMuons, simTraksHandle.product(), simVertices.product(), simTrackFilter);
    }
    //todo something when noMatcher

  } else if (edmCfg.exists("trackingParticleTag")) {
    edm::Handle<TrackingParticleCollection> trackingParticleHandle;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("trackingParticleTag"), trackingParticleHandle);
    LogTrace("l1tOmtfEventPrint") << "\nCandidateSimMuonMatcher::observeEventEnd trackingParticleHandle size "
                                  << trackingParticleHandle.product()->size() << std::endl;

    //TODO use other trackParticleFilter if needed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //TrackingParticle is used for minBias sample. There we should do propagation to match pions and kaons
    std::function<bool(const TrackingParticle&)> trackParticleFilter = trackingParticleIsMuonInBx0Displ;
    //trackingParticleIsMuonInBx0; if propagation is used, use trackingParticleIsMuonInOmtfBx0
    matchingResults = match(finalMuons, trackingParticleHandle.product(), trackParticleFilter);
  } else if (matchingType == MatchingType::collectMuonCands) {
    matchingResults = collectMuonCands(finalMuons);
  }
}

void CandidateSimMuonMatcher::endJob() {}

FinalMuons CandidateSimMuonMatcher::ghostBust(const FinalMuons& finalMuons) {
  boost::dynamic_bitset<> isKilled(finalMuons.size(), false);
  //finalMuons are nonempty
  for (unsigned int i1 = 0; i1 < finalMuons.size(); ++i1) {
    LogTrace("l1tOmtfEventPrint") << "\nCandidateSimMuonMatcher::ghostBusting\n" << *(finalMuons[i1]);
    if (finalMuons[i1]->getPtGmt() <= 0) {
      //PtGmt > 0 marks valid candidate both for phase1 and phase2. But it should not matter here, as all finalMuons are not empty, so it just a safety check
      throw cms::Exception("CandidateSimMuonMatcher")
          << "CandidateSimMuonMatcher::ghostBust finalMuons[" << i1 << "] has ptGmt <= 0, ptGmt "
          << finalMuons[i1]->getPtGmt() << std::endl;
    }
    for (unsigned int i2 = i1 + 1; i2 < finalMuons.size(); ++i2) {
      auto& mtfCand1 = finalMuons[i1];
      auto& mtfCand2 = finalMuons[i2];
      //if (mtfCand2->getPt() == 0)
      //  continue;

      if (std::abs(mtfCand1->getEtaRad() - mtfCand2->getEtaRad()) < 0.3) {
        //folding phi
        double deltaPhi = std::abs(mtfCand1->getPhiRad() - mtfCand2->getPhiRad());
        if (deltaPhi > M_PI)
          deltaPhi = std::abs(deltaPhi - 2 * M_PI);

        //0.116355283466 is 10 units of the phase1
        if (deltaPhi < 10 * 2 * M_PI / 576) {
          if (mtfCand1->getFiredLayerCnt() > mtfCand2->getFiredLayerCnt()) {
            isKilled[i2] = true;
          } else
            isKilled[i1] = true;
        }
        LogTrace("l1tOmtfEventPrint") << " mtfCand2 PhiRad " << mtfCand2->getPhiRad() << " etaRad "
                                      << mtfCand2->getEtaRad() << " deltaPhi " << deltaPhi << " isKilled[i2] "
                                      << isKilled[i2] << " isKilled[i1] " << isKilled[i1];
      }
    }
  }

  FinalMuons resultCands;

  LogTrace("l1tOmtfEventPrint") << "\nCandidateSimMuonMatcher::ghostBust - after ghostBusting:" << std::endl;
  for (unsigned int i1 = 0; i1 < finalMuons.size(); ++i1) {
    //not dropping candidates with quality 0 and 1
    if (!isKilled[i1]) {  //&& finalMuons[i1]->getQuality() > 1
      resultCands.push_back(finalMuons[i1]);
    }

    //if (finalMuons[i1]->getPtGmt() > 0) checked earlier, so it is not needed here
    {  //PtGmt > 0 marks valid candidate both for phase1 and phase2. But it should not matter here, as all finalMuons are not empty

      LogTrace("l1tOmtfEventPrint") << *(finalMuons[i1]) << " gb isKilled " << isKilled.test(i1) << std::endl;
      //LogTrace("l1tOmtfEventPrint") << *(gbCandidates.at(i1)) << std::endl;
      if (finalMuons[i1]->getAlgoMuon())
        LogTrace("l1tOmtfEventPrint") << *(finalMuons[i1]->getAlgoMuon()) << std::endl;
      LogTrace("l1tOmtfEventPrint") << std::endl;
    }
  }

  if (resultCands.size() >= 3)
    LogTrace("l1tOmtfEventPrint") << " ghost !!!!!! " << std::endl;
  LogTrace("l1tOmtfEventPrint") << std::endl;

  return resultCands;
}

TrajectoryStateOnSurface CandidateSimMuonMatcher::atStation1(const FreeTrajectoryState& ftsStart) const {
  // propagate to MB1, which defines the OMTF region (W+-2 MB1 is connected only to the OMTF)
  // 415 cm is R of RB1in, 660.5cm is |z| of the edge of MB2 (B field on)
  ReferenceCountingPointer<Surface> rpc = ReferenceCountingPointer<Surface>(new BoundCylinder(
      GlobalPoint(0., 0., 0.), TkRotation<float>(), SimpleCylinderBounds(431.133, 431.133, -660.5, 660.5)));
  //N.B. zMin and zMax do not matter for the propagator->propagate, i.e. there is not cut on them
  TrajectoryStateOnSurface trackAtRPC = propagator->propagate(ftsStart, *rpc);

  return trackAtRPC;
}

TrajectoryStateOnSurface CandidateSimMuonMatcher::atStation2(const FreeTrajectoryState& ftsStart) const {
  ReferenceCountingPointer<Surface> surface = ReferenceCountingPointer<Surface>(new BoundCylinder(
      GlobalPoint(0., 0., 0.), TkRotation<float>(), SimpleCylinderBounds(512.401, 512.401, -900, 900)));

  TrajectoryStateOnSurface tsof = propagator->propagate(ftsStart, *surface);
  return tsof;
}

FreeTrajectoryState CandidateSimMuonMatcher::simTrackToFts(const SimTrack& simTrackPtr, const SimVertex& simVertex) {
  int charge = simTrackPtr.charge();  //simTrackPtr.type() > 0 ? -1 : 1;  //works for muons

  GlobalVector p3GV(simTrackPtr.momentum().x(), simTrackPtr.momentum().y(), simTrackPtr.momentum().z());
  GlobalPoint r3GP(simVertex.position().x(), simVertex.position().y(), simVertex.position().z());

  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, &*magField);

  return FreeTrajectoryState(tPars);
}

FreeTrajectoryState CandidateSimMuonMatcher::simTrackToFts(const TrackingParticle& trackingParticle) {
  int charge = trackingParticle.charge();  //trackingParticle.pdgId() > 0 ? -1 : 1;  //works for muons

  GlobalVector p3GV(trackingParticle.momentum().x(), trackingParticle.momentum().y(), trackingParticle.momentum().z());
  GlobalPoint r3GP(trackingParticle.vx(), trackingParticle.vy(), trackingParticle.vz());

  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, &*magField);

  return FreeTrajectoryState(tPars);
}

TrajectoryStateOnSurface CandidateSimMuonMatcher::propagate(const SimTrack& simTrack,
                                                            const edm::SimVertexContainer* simVertices) {
  SimVertex simVertex;
  int vtxInd = simTrack.vertIndex();
  if (vtxInd < 0) {
    edm::LogImportant("l1tOmtfEventPrint") << "Track with no vertex, defaulting to (0,0,0)";
  } else {
    simVertex = simVertices->at(vtxInd);
    if (((int)simVertex.vertexId()) != vtxInd) {
      edm::LogImportant("l1tOmtfEventPrint") << "simVertex.vertexId() != vtxInd. simVertex.vertexId() "
                                             << simVertex.vertexId() << " vtxInd " << vtxInd << " !!!!!!!!!!!!!!!!!";
      throw cms::Exception("CandidateSimMuonMatcher",
                           "CandidateSimMuonMatcher::propagate: simVertex.vertexId() != vtxInd");
    }
  }

  FreeTrajectoryState ftsTrack = simTrackToFts(simTrack, simVertex);

  TrajectoryStateOnSurface tsof = atStation2(ftsTrack);  //propagation

  return tsof;
}

TrajectoryStateOnSurface CandidateSimMuonMatcher::propagate(const TrackingParticle& trackingParticle) {
  FreeTrajectoryState ftsTrack = simTrackToFts(trackingParticle);

  TrajectoryStateOnSurface tsof = atStation2(ftsTrack);  //propagation

  return tsof;
}

float normal_pdf(float x, float m, float s) {
  static const float inv_sqrt_2pi = 0.3989422804014327;
  float a = (x - m) / s;

  return inv_sqrt_2pi / s * std::exp(-0.5 * a * a);
}

void CandidateSimMuonMatcher::match(const FinalMuonPtr& finalMuon, MatchingResult& result) {
  double trackPt = result.genPt;

  //if (std::abs(simTrack.momentum().eta() - candGloablEta) < 0.3) //has no sense for displaced muons
  {
    result.muonCand = finalMuon;

    if (matchingType == MatchingType::simplePropagation) {
      //int charge = result.pdgId > 0 ? -1 : 1;  //works for muons

      TH1* medianDelta = nullptr;
      TH1* minDelta = nullptr;
      TH1* maxDelta = nullptr;

      if (result.genCharge == 1) {
        medianDelta = medianDelta_pos;
        minDelta = minDelta_pos;
        maxDelta = maxDelta_pos;
      } else {
        medianDelta = medianDelta_neg;
        minDelta = minDelta_neg;
        maxDelta = maxDelta_neg;
      }

      auto ptBin = medianDelta->FindBin(trackPt);
      auto min = minDelta->GetBinContent(ptBin);
      auto max = maxDelta->GetBinContent(ptBin);

      result.propagatedPhi = foldPhi(result.genPhi + medianDelta->GetBinContent(ptBin));

      result.deltaPhi = foldPhi(result.genPhi - finalMuon->getPhiRad());
      result.deltaEta = result.propagatedEta - finalMuon->getEtaRad();

      result.matchingLikelihood = 1. / (std::abs(result.deltaPhi) + 0.001);

      if (result.deltaPhi > min && result.deltaPhi < max && std::abs(result.deltaEta) < 0.4)
        result.result = MatchingResult::ResultType::matched;
    } else if (matchingType == MatchingType::withPropagator) {
      result.deltaPhi = foldPhi(result.propagatedPhi - finalMuon->getPhiRad());
      result.deltaEta = result.propagatedEta - finalMuon->getEtaRad();  //propagatedEta is set in propagate

      result.matchingLikelihood = 1. / (std::abs(result.deltaPhi) + 0.001);

      double mean = 0;
      double treshold = 0;

      //for displaced muons in H2ll
      //here a coarse threshold are used, as for dispalced muons
      treshold = 0.15;   //pt > 30
      if (trackPt < 10)  //TODO!!!!!!!!!!!!!!!!!!!!! tune the threshold!!!!!!
        treshold = 0.4;
      else if (trackPt < 30)  //TODO!!!!!!!!!!!!!!! tune the threshold!!!!!!
        treshold = 0.22;

      mean = 0;

      if (std::abs(result.deltaPhi - mean) < treshold && std::abs(result.deltaEta) < 0.4)
        result.result = MatchingResult::ResultType::matched;
    }

    LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::match::" << __LINE__ << "\n" << result << std::endl;
  }
}

std::vector<MatchingResult> CandidateSimMuonMatcher::cleanMatching(std::vector<MatchingResult> matchingResults,
                                                                   const FinalMuons& finalMuons) {
  //Cleaning the matching
  std::sort(
      matchingResults.begin(), matchingResults.end(), [](const MatchingResult& a, const MatchingResult& b) -> bool {
        return a.matchingLikelihood > b.matchingLikelihood;
      });

  for (unsigned int i1 = 0; i1 < matchingResults.size(); i1++) {
    if (matchingResults[i1].result == MatchingResult::ResultType::matched) {
      for (unsigned int i2 = i1 + 1; i2 < matchingResults.size(); i2++) {
        if ((matchingResults[i1].trackingParticle &&
             matchingResults[i1].trackingParticle == matchingResults[i2].trackingParticle) ||
            (matchingResults[i1].simTrack && matchingResults[i1].simTrack == matchingResults[i2].simTrack) ||
            (matchingResults[i1].muonCand == matchingResults[i2].muonCand)) {
          //if matchingResults[i1].muonCand == false, then it is also OK here
          matchingResults[i2].result = MatchingResult::ResultType::duplicate;
        }
      }
    }
  }

  std::vector<MatchingResult> cleanedMatchingResults;
  for (auto& matchingResult : matchingResults) {
    if (matchingResult.result == MatchingResult::ResultType::matched || matchingResult.muonCand == nullptr)
      //adding also the simTracks that are not matched at all, before it is assured that they are not duplicates
      cleanedMatchingResults.push_back(matchingResult);
  }

  //adding the muonCand-s that were not matched, e.g. in order to analyze them later
  for (auto& muonCand : finalMuons) {
    //PtGmt > 0 marks valid candidate both for phase1 and phase2. But this check is done in ghostBust
    //if(muonCand->getPtGmt() == 0) // || muonCand->getQuality() == 0
    //  continue;

    bool isMatched = false;
    for (auto& matchingResult : cleanedMatchingResults) {
      if (matchingResult.muonCand == muonCand) {
        isMatched = true;
        break;
      }
    }

    if (!isMatched) {
      MatchingResult result;
      result.muonCand = muonCand;
      cleanedMatchingResults.push_back(result);
    }
  }

  LogTrace("l1tOmtfEventPrint") << "\nCandidateSimMuonMatcher::cleanMatching:" << __LINE__
                                << " cleanedMatchingResults:" << std::endl;
  for (auto& result : cleanedMatchingResults) {
    LogTrace("l1tOmtfEventPrint") << result << std::endl << std::endl;
  }
  LogTrace("l1tOmtfEventPrint") << " " << std::endl;

  return cleanedMatchingResults;
}

void CandidateSimMuonMatcher::propagate(MatchingResult& result) {
  auto doSimplePropagation = [&]() {
    TH1* medianDelta = result.genCharge == 1 ? medianDelta_pos : medianDelta_neg;
    auto ptBin = medianDelta->FindBin(result.genPt);
    result.propagatedPhi = foldPhi(result.genPhi + medianDelta->GetBinContent(ptBin));
    result.propagatedEta = result.genEta;
  };

  if (matchingType == MatchingType::withPropagator) {
    FreeTrajectoryState ftsTrack;
    if (result.simTrack) {
      if (result.simVertex) {
        ftsTrack = simTrackToFts(*(result.simTrack), *(result.simVertex));
      } else {
        throw cms::Exception("CandidateSimMuonMatcher")
            << "CandidateSimMuonMatcher::propagate - simTrack exists but simVertex is missing!!!";
        //throw, to be sue that the issue is noticed.
        // Alternatively, if simTrack exists but simVertex is missing,
        // fall back to a default vertex at (0,0,0). Or doSimplePropagation().
        SimVertex defVtx;  // default constructed vertex at origin
        ftsTrack = simTrackToFts(*(result.simTrack), defVtx);
        edm::LogImportant("l1tOmtfEventPrint")
            << "CandidateSimMuonMatcher::propagate - simTrack has no simVertex, using default vertex" << std::endl;
      }
    } else if (result.trackingParticle) {
      if (result.trackingParticle->parentVertex().isNonnull())
        ftsTrack = simTrackToFts(*result.trackingParticle);
      else {
        edm::LogImportant("l1tOmtfEventPrint")
            << "CandidateSimMuonMatcher::propagate - trackingParticle has no parentVertex!!!" << std::endl;
        throw cms::Exception("CandidateSimMuonMatcher")
            << "CandidateSimMuonMatcher::propagate - trackingParticle has no parentVertex!!!";
      }
    } else {
      // Nothing to propagate
      result.result = MatchingResult::ResultType::propagationFailed;
      return;
    }
    //TODO check if the ftsTrack is valid
    TrajectoryStateOnSurface tsof = atStation2(ftsTrack);  //propagation

    if (!tsof.isValid()) {
      if (result.genPt > 2.5 && result.trackingParticle && result.trackingParticle->parentVertex().isNonnull()) {
        if (result.muonRho < 40) {  //TODO why it is done only for trackingParticle and not for simTrack?
          doSimplePropagation();
        }
      } else {
        result.result = MatchingResult::ResultType::propagationFailed;
        //result.propagatedEta = 0 in this case
      }
    } else {
      result.propagatedPhi = tsof.globalPosition().phi();
      result.propagatedEta = tsof.globalPosition().eta();
    }
  } else if (matchingType == MatchingType::simplePropagation) {
    doSimplePropagation();
  }
}

void CandidateSimMuonMatcher::match(const FinalMuons& finalMuons,
                                    MatchingResult& result,
                                    std::vector<MatchingResult>& matchingResults) {
  propagate(result);
  if (result.result == MatchingResult::ResultType::propagationFailed) {  //no sense to do matching

    //TODO For the displaced muons adding the muons for which the propagation failed in principle has no sense
    //as matching with candidates is not possible then.
    //For prompt muons this can be useful, to have the full pt spectrum of gen muons.
    //However, using them in the denominator of the efficiency biases the efficiency, because for these muons matching to the candidates is not possible.
    //In any case these results are marked in the DataROOTDumper2 omtfEvent.muonEvent = -2.
    LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " propagation failed: genPt " << result.genPt
                                  << " genEta " << result.genEta << " eventId "  //<< simTrack.eventId().event()
                                  << std::endl;

    //matchingResults.push_back(result);
  } else {
    //checking if the propagated track is inside the OMTF range, TODO - tune the range!!!!!!!!!!!!!!!!!
    //eta 0.7 is the beginning of the MB2,
    //the eta range wider than the nominal OMTF region is needed, as in any case muons outside this region are seen by the OMTF
    //so it is better to match them to simMuon, otherwise they look like ghosts.
    //Besides, it better to train the nn suich that is able to measure its pt, as it may affect the rate
    if ((std::abs(result.propagatedEta) >= 0.7) && (std::abs(result.propagatedEta) <= 1.31)) {
      LogTrace("l1tOmtfEventPrint")
          << "CandidateSimMuonMatcher::match simTrack IS in OMTF region, matching to the omtfCands, propagatedEta: "
          << result.propagatedEta;
    } else {
      LogTrace("l1tOmtfEventPrint") << "simTrack NOT in OMTF region ";
      return;
    }

    /* TODO fix if filling of the deltaPhiPropCandMean and deltaPhiPropCandStdDev is needed
      double ptGen = simTrack.momentum().pt();
      if(ptGen >= deltaPhiVertexProp->GetXaxis()->GetXmax())
        ptGen = deltaPhiVertexProp->GetXaxis()->GetXmax() - 0.01;

      deltaPhiVertexProp->Fill(ptGen, simTrack.momentum().phi() - tsof.globalPosition().phi());*/

    unsigned int iCand = 0;
    bool matched = false;
    for (auto& muonCand : finalMuons) {
      //dropping very low quality candidates, as they are fakes usually - but it has no sense, then the results are not conclusive
      //if (muonCand->getQuality() > 1)
      {  //TOOD uncomment, if this condition is needed
        MatchingResult resultCopy = result;

        match(muonCand, resultCopy);

        if (resultCopy.result == MatchingResult::ResultType::matched) {
          matchingResults.push_back(resultCopy);
          matched = true;
        }
      }
      iCand++;
    }

    if (!matched) {  //adding matchingResults (i.e. simMuon in this case) also if it was not matched to any candidate
      matchingResults.push_back(result);
      LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " no matching candidate found" << std::endl;
    }
  }
}

std::vector<MatchingResult> CandidateSimMuonMatcher::match(const FinalMuons& finalMuons,
                                                           const edm::SimTrackContainer* simTracks,
                                                           const edm::SimVertexContainer* simVertices,
                                                           std::function<bool(const SimTrack&)> const& simTrackFilter) {
  std::vector<MatchingResult> matchingResults;

  for (auto& simTrack : *simTracks) {
    if (!simTrackFilter(simTrack))
      continue;

    LogTrace("l1tOmtfEventPrint") << "\nCandidateSimMuonMatcher::match, simTrack type " << std::setw(3)
                                  << simTrack.type() << " pt " << std::setw(9) << simTrack.momentum().pt()
                                  << " GeV eta " << std::setw(9) << simTrack.momentum().eta() << " phi " << std::setw(9)
                                  << simTrack.momentum().phi() << " rho "
                                  << (simTrack.vertIndex() >= 0 ? simVertices->at(simTrack.vertIndex()).position().Rho()
                                                                : -99)
                                  << std::endl;

    SimVertex simVertex;
    int vtxInd = simTrack.vertIndex();
    if (vtxInd < 0) {
      edm::LogImportant("l1tOmtfEventPrint") << "Track with no vertex, defaulting to (0,0,0)";
    } else {
      simVertex = simVertices->at(vtxInd);
      if (((int)simVertex.vertexId()) != vtxInd) {
        edm::LogImportant("l1tOmtfEventPrint") << "simVertex.vertexId() != vtxInd. simVertex.vertexId() "
                                               << simVertex.vertexId() << " vtxInd " << vtxInd << " !!!!!!!!!!!!!!!!!";
      }
    }

    MatchingResult result(simTrack, simTrack.vertIndex() >= 0 ? &(simVertices->at(simTrack.vertIndex())) : nullptr);

    match(finalMuons, result, matchingResults);
  }

  return cleanMatching(matchingResults, finalMuons);
}

std::vector<MatchingResult> CandidateSimMuonMatcher::match(
    const FinalMuons& finalMuons,
    const TrackingParticleCollection* trackingParticles,
    std::function<bool(const TrackingParticle&)> const& simTrackFilter) {
  std::vector<MatchingResult> matchingResults;
  LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::match trackingParticles->size() "
                                << trackingParticles->size() << std::endl;

  for (auto& trackingParticle : *trackingParticles) {
    //LogTrace("l1tOmtfEventPrint") <<"CandidateSimMuonMatcher::match:"<<__LINE__<<" trackingParticle type "<<std::setw(3)<<trackingParticle.pdgId()<<" pt "<<std::setw(9)<<trackingParticle.pt()<<" eta "<<std::setw(9)<<trackingParticle.momentum().eta()<<" phi "<<std::setw(9)<<trackingParticle.momentum().phi()<<std::endl;

    if (simTrackFilter(trackingParticle) == false)
      continue;

    LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::match, trackingParticle type " << std::setw(3)
                                  << trackingParticle.pdgId() << " pt " << std::setw(9) << trackingParticle.pt()
                                  << " GeV eta " << std::setw(9) << trackingParticle.momentum().eta() << " phi "
                                  << std::setw(9) << trackingParticle.momentum().phi() << std::endl;

    MatchingResult result(trackingParticle);

    match(finalMuons, result, matchingResults);
  }

  return cleanMatching(matchingResults, finalMuons);
}

std::vector<MatchingResult> CandidateSimMuonMatcher::matchSimple(
    const FinalMuons& finalMuons,
    const edm::SimTrackContainer* simTracks,
    const edm::SimVertexContainer* simVertices,
    std::function<bool(const SimTrack&)> const& simTrackFilter) {
  std::vector<MatchingResult> matchingResults;

  for (auto& simTrack : *simTracks) {
    if (!simTrackFilter(simTrack))
      continue;

    LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::matchSimple: simTrack type " << std::setw(3)
                                  << simTrack.type() << " pt " << std::setw(9) << simTrack.momentum().pt()
                                  << " GeV eta " << std::setw(9) << simTrack.momentum().eta() << " phi " << std::setw(9)
                                  << simTrack.momentum().phi() << std::endl;

    bool matched = false;

    for (auto& finalMuon : finalMuons) {
      //dropping very low quality candidates, as they are fakes usually - but it has no sense, then the results are not conclusive
      //if(muonCand->hwQual() > 1)
      //if (finalMuon->getPtGev() > 0) in phase-1 ptGeV=0 is possible for a valid candidate
      {
        MatchingResult result(simTrack, simTrack.vertIndex() >= 0 ? &(simVertices->at(simTrack.vertIndex())) : nullptr);

        result.deltaPhi = foldPhi(result.genPhi - finalMuon->getPhiRad());
        result.deltaEta = result.genEta - finalMuon->getEtaRad();

        result.propagatedPhi = result.genPhi;
        result.propagatedEta = result.genEta;

        result.muonCand = finalMuon;

        //TODO histogram can be used, like in the usercode/L1MuonAnalyzer  MuonMatcher::matchWithoutPorpagation
        //for prompt muons
        /*double treshold = 0.3;
        if (simTrack.momentum().pt() < 5)
          treshold = 1.5;
        else if (simTrack.momentum().pt() < 8)
          treshold = 0.8;
        else if (simTrack.momentum().pt() < 10)
          treshold = 0.6;
        else if (simTrack.momentum().pt() < 20)
          treshold = 0.5;*/

        double treshold = 0.5;
        if (simTrack.momentum().pt() < 5)
          treshold = 1.5;
        else if (simTrack.momentum().pt() < 10)
          treshold = 1.0;
        else if (simTrack.momentum().pt() < 25)
          treshold = 0.7;

        if (std::abs(result.deltaPhi) < treshold && std::abs(result.deltaEta) < 0.5) {
          result.result = MatchingResult::ResultType::matched;
          //matchingLikelihood is needed in the cleanMatching, so we put something
          result.matchingLikelihood = 1. / (std::abs(result.deltaPhi) + 0.001);
        }

        LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::matchSimple:\n" << result << std::endl;

        if (result.result == MatchingResult::ResultType::matched) {
          matchingResults.push_back(result);
          matched = true;
        }
      }
    }

    if (!matched) {  //we are adding also if it was not matched to any candidate
      MatchingResult result(simTrack, simTrack.vertIndex() >= 0 ? &(simVertices->at(simTrack.vertIndex())) : nullptr);
      result.propagatedPhi = result.genPhi;
      result.propagatedEta = result.genEta;
      matchingResults.push_back(result);
      LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " no matching candidate found" << std::endl;
    }
  }

  return cleanMatching(matchingResults, finalMuons);
}

std::vector<MatchingResult> CandidateSimMuonMatcher::collectMuonCands(const FinalMuons& finalMuons) {
  std::vector<MatchingResult> matchingResults;
  unsigned int iCand = 0;
  for (auto& finalMuon : finalMuons) {
    //dropping very low quality candidates, as they are fakes usually - but it has no sense, then the results are not conclusive
    //if(muonCand->hwQual() > 1)
    {
      MatchingResult result;

      result.muonCand = finalMuon;

      LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::collectMuonCands: " << *finalMuon << std::endl;

      matchingResults.push_back(result);
    }
    iCand++;
  }
  return matchingResults;
}
