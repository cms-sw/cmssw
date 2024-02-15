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

double hwGmtPhiToGlobalPhi(int phi) {
  double phiGmtUnit = 2. * M_PI / 576.;
  return phi * phiGmtUnit;
}

double foldPhi(double phi) {
  if (phi > M_PI)
    return (phi - 2 * M_PI);
  else if (phi < -M_PI)
    return (phi + 2 * M_PI);

  return phi;
}

CandidateSimMuonMatcher::CandidateSimMuonMatcher(
    const edm::ParameterSet& edmCfg,
    const OMTFConfiguration* omtfConfig,
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
    const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken)
    : omtfConfig(omtfConfig),
      edmCfg(edmCfg),
      magneticFieldEsToken(magneticFieldEsToken),
      propagatorEsToken(propagatorEsToken) {
  std::string muonMatcherFileName = edmCfg.getParameter<edm::FileInPath>("muonMatcherFile").fullPath();
  TFile inFile(muonMatcherFileName.c_str());
  edm::LogImportant("l1tOmtfEventPrint") << " CandidateSimMuonMatcher: using muonMatcherFileName "
                                         << muonMatcherFileName << std::endl;

  deltaPhiPropCandMean = (TH1D*)inFile.Get("deltaPhiPropCandMean");
  deltaPhiPropCandStdDev = (TH1D*)inFile.Get("deltaPhiPropCandStdDev");
}

CandidateSimMuonMatcher::~CandidateSimMuonMatcher() {}

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
                                                       const std::vector<l1t::RegionalMuonCand>& candMuons) {
  //debug
  unsigned int procIndx = omtfConfig->getProcIndx(iProcessor, mtfType);
  for (auto& gbCandidate : gbCandidates) {
    if (gbCandidate->getPtConstr() > 0) {
      LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::observeProcesorEmulation procIndx" << procIndx << " "
                                    << *gbCandidate << std::endl;
      this->gbCandidates.emplace_back(gbCandidate);
    }
  }
}

bool simTrackIsMuonInOmtf(const SimTrack& simTrack) {
  if (std::abs(simTrack.type()) == 13 ||
      std::abs(simTrack.type()) == 1000015) {  // 1000015 is stau, todo use other selection (e.g. pt>20) if needed
    //only muons
  } else
    return false;

  //in the overlap, the propagation of muons with pt less then ~3.2 fails - the actual threshold depends slightly on eta,
  if (simTrack.momentum().pt() < 2.5)
    return false;

  LogTrace("l1tOmtfEventPrint") << "simTrackIsMuonInOmtf, simTrack type " << std::setw(3) << simTrack.type() << " pt "
                                << std::setw(9) << simTrack.momentum().pt() << " eta " << std::setw(9)
                                << simTrack.momentum().eta() << " phi " << std::setw(9) << simTrack.momentum().phi()
                                << std::endl;

  //some margin for matching must be used on top of actual OMTF region,
  //i.e. (0.82-1.24)=>(0.72-1.3),
  //otherwise many candidates are marked as ghosts
  if ((std::abs(simTrack.momentum().eta()) >= 0.72) && (std::abs(simTrack.momentum().eta()) <= 1.3)) {
  } else
    return false;

  return true;
}

bool simTrackIsMuonInOmtfBx0(const SimTrack& simTrack) {
  if (simTrack.eventId().bunchCrossing() != 0)
    return false;

  return simTrackIsMuonInOmtf(simTrack);
}

bool simTrackIsMuonInBx0(const SimTrack& simTrack) {
  if (std::abs(simTrack.type()) == 13 ||
      std::abs(simTrack.type()) == 1000015) {  // 1000015 is stau, todo use other selection (e.g. pt>20) if needed
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
  if ((std::abs(trackingParticle.momentum().eta()) >= 0.72) && (std::abs(trackingParticle.momentum().eta()) <= 1.3)) {
  } else
    return false;

  return true;
}

bool trackingParticleIsMuonInOmtfEvent0(const TrackingParticle& trackingParticle) {
  if (trackingParticle.eventId().event() != 0)
    return false;

  return trackingParticleIsMuonInOmtfBx0(trackingParticle);
}

void CandidateSimMuonMatcher::observeEventEnd(const edm::Event& event,
                                              std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) {
  LogTrace("l1tOmtfEventPrint") << "\nCandidateSimMuonMatcher::observeEventEnd" << std::endl;
  AlgoMuons ghostBustedProcMuons;
  std::vector<const l1t::RegionalMuonCand*> ghostBustedRegionalCands =
      CandidateSimMuonMatcher::ghostBust(finalCandidates.get(), gbCandidates, ghostBustedProcMuons);

  matchingResults.clear();
  if (edmCfg.exists("simTracksTag")) {
    edm::Handle<edm::SimTrackContainer> simTraksHandle;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("simTracksTag"), simTraksHandle);

    edm::Handle<edm::SimVertexContainer> simVertices;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("simVertexesTag"), simVertices);

    LogTrace("l1tOmtfEventPrint") << "simTraksHandle size " << simTraksHandle.product()->size() << std::endl;
    LogTrace("l1tOmtfEventPrint") << "simVertices size " << simVertices.product()->size() << std::endl;

    //TODO  use other simTrackFilter if needed  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //we dont want to check the eta of the generated muon, as it is on the vertex,
    //instead inside match, we check the eta of the propagated track to the second muons station
    std::function<bool(const SimTrack&)> const& simTrackFilter = simTrackIsMuonInBx0;  //simTrackIsMuonInOmtfBx0;

    matchingResults = match(
        ghostBustedRegionalCands, ghostBustedProcMuons, simTraksHandle.product(), simVertices.product(), simTrackFilter);

  } else if (edmCfg.exists("trackingParticleTag")) {
    edm::Handle<TrackingParticleCollection> trackingParticleHandle;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("trackingParticleTag"), trackingParticleHandle);
    LogTrace("l1tOmtfEventPrint") << "\nCandidateSimMuonMatcher::observeEventEnd trackingParticleHandle size "
                                  << trackingParticleHandle.product()->size() << std::endl;

    //TODO use other trackParticleFilter if needed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    std::function<bool(const TrackingParticle&)> trackParticleFilter =
        trackingParticleIsMuonInBx0;  //trackingParticleIsMuonInOmtfBx0;
    matchingResults =
        match(ghostBustedRegionalCands, ghostBustedProcMuons, trackingParticleHandle.product(), trackParticleFilter);
  }
}

void CandidateSimMuonMatcher::endJob() {}

std::vector<const l1t::RegionalMuonCand*> CandidateSimMuonMatcher::ghostBust(
    const l1t::RegionalMuonCandBxCollection* mtfCands, const AlgoMuons& gbCandidates, AlgoMuons& ghostBustedProcMuons) {
  if (gbCandidates.size() != mtfCands->size(0)) {
    edm::LogError("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::ghostBust(): gbCandidates.size() "
                                       << gbCandidates.size() << " != mtfCands.size() " << mtfCands->size();
  }

  boost::dynamic_bitset<> isKilled(mtfCands->size(0), false);

  for (unsigned int i1 = 0; i1 < mtfCands->size(0); ++i1) {
    if (mtfCands->at(0, i1).hwPt() == 0)
      continue;
    LogTrace("l1tOmtfEventPrint") << "\nCandidateSimMuonMatcher::ghostBust regionalCand pt " << std::setw(3)
                                  << mtfCands->at(0, i1).hwPt() << " qual " << std::setw(2)
                                  << mtfCands->at(0, i1).hwQual() << " proc " << std::setw(2)
                                  << mtfCands->at(0, i1).processor();
    for (unsigned int i2 = i1 + 1; i2 < mtfCands->size(0); ++i2) {
      auto& mtfCand1 = mtfCands->at(0, i1);
      auto& mtfCand2 = mtfCands->at(0, i2);
      if (mtfCand2.hwPt() == 0)
        continue;

      if (std::abs(mtfCand1.hwEta() - mtfCand2.hwEta()) < (0.3 / 0.010875)) {
        int gloablHwPhi1 = omtfConfig->calcGlobalPhi(mtfCand1.hwPhi(), mtfCand1.processor());
        int gloablHwPhi2 = omtfConfig->calcGlobalPhi(mtfCand2.hwPhi(), mtfCand2.processor());

        //one can use the phi in radians like that:
        //double globalPhi1 = hwGmtPhiToGlobalPhi(omtfConfig->calcGlobalPhi( mtfCand1.hwPhi(), mtfCand1.processor() ) );
        //double globalPhi2 = hwGmtPhiToGlobalPhi(omtfConfig->calcGlobalPhi( mtfCand2.hwPhi(), mtfCand2.processor() ) );

        //0.0872664626 = 5 deg, i.e. the same window as in the OMTF ghost buster
        if (std::abs(gloablHwPhi1 - gloablHwPhi2) < 8) {
          //if (mtfCand1.hwQual() > mtfCand2.hwQual()) //TODO this is used in the uGMT
          if (gbCandidates[i1]->getFiredLayerCnt() >
              gbCandidates[i2]->getFiredLayerCnt())  //but this should be better - but probably the difference is not big
          {
            isKilled[i2] = true;
          } else
            isKilled[i1] = true;
        }
      }
    }
  }

  std::vector<const l1t::RegionalMuonCand*> resultCands;

  for (unsigned int i1 = 0; i1 < mtfCands->size(0); ++i1) {
    //dropping candidates with quality 0 !!!!!!!!!!!!!!!!!!!! fixme if not needed
    if (!isKilled[i1] && mtfCands->at(0, i1).hwPt()) {
      resultCands.push_back(&(mtfCands->at(0, i1)));
      ghostBustedProcMuons.push_back(gbCandidates.at(i1));
    }

    if (mtfCands->at(0, i1).hwPt()) {
      LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::ghostBust\n  regionalCand pt " << std::setw(3)
                                    << mtfCands->at(0, i1).hwPt() << " qual " << std::setw(2)
                                    << mtfCands->at(0, i1).hwQual() << " proc " << std::setw(2)
                                    << mtfCands->at(0, i1).processor() << " eta " << std::setw(4)
                                    << mtfCands->at(0, i1).hwEta() << " gloablEta " << std::setw(8)
                                    << mtfCands->at(0, i1).hwEta() * 0.010875 << " hwPhi " << std::setw(3)
                                    << mtfCands->at(0, i1).hwPhi() << " globalPhi " << std::setw(8)
                                    << hwGmtPhiToGlobalPhi(omtfConfig->calcGlobalPhi(mtfCands->at(0, i1).hwPhi(),
                                                                                     mtfCands->at(0, i1).processor()))
                                    << " fireadLayers " << std::bitset<18>(mtfCands->at(0, i1).trackAddress().at(0))
                                    << " gb isKilled " << isKilled.test(i1) << std::endl;

      LogTrace("l1tOmtfEventPrint") << *(gbCandidates.at(i1)) << std::endl;
    }
  }

  if (resultCands.size() >= 3)
    LogTrace("l1tOmtfEventPrint") << " ghost !!!!!! " << std::endl;
  LogTrace("l1tOmtfEventPrint") << std::endl;

  return resultCands;
}

TrajectoryStateOnSurface CandidateSimMuonMatcher::atStation2(FreeTrajectoryState ftsStart, float eta) const {
  eta = 0;  //fix me!!!!! in case of displaced muon the vertex eta has no sense
  ReferenceCountingPointer<Surface> rpc;
  if (eta < -1.24)  //negative endcap, RE2
    rpc = ReferenceCountingPointer<Surface>(
        new BoundDisk(GlobalPoint(0., 0., -790.), TkRotation<float>(), SimpleDiskBounds(300., 810., -10., 10.)));
  else if (eta < 1.24)  //barrel + overlap, 512.401cm is R of middle of the MB2
    rpc = ReferenceCountingPointer<Surface>(new BoundCylinder(
        GlobalPoint(0., 0., 0.), TkRotation<float>(), SimpleCylinderBounds(512.401, 512.401, -900, 900)));
  else
    rpc = ReferenceCountingPointer<Surface>(  //positive endcap, RE2
        new BoundDisk(GlobalPoint(0., 0., 790.), TkRotation<float>(), SimpleDiskBounds(300., 810., -10., 10.)));

  TrajectoryStateOnSurface trackAtRPC = propagator->propagate(ftsStart, *rpc);
  return trackAtRPC;
}

FreeTrajectoryState CandidateSimMuonMatcher::simTrackToFts(const SimTrack& simTrackPtr, const SimVertex& simVertex) {
  int charge = simTrackPtr.type() > 0 ? -1 : 1;  //works for muons

  GlobalVector p3GV(simTrackPtr.momentum().x(), simTrackPtr.momentum().y(), simTrackPtr.momentum().z());
  GlobalPoint r3GP(simVertex.position().x(), simVertex.position().y(), simVertex.position().z());

  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, &*magField);

  return FreeTrajectoryState(tPars);
}

FreeTrajectoryState CandidateSimMuonMatcher::simTrackToFts(const TrackingParticle& trackingParticle) {
  int charge = trackingParticle.pdgId() > 0 ? -1 : 1;  //works for muons

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
    }
  }

  FreeTrajectoryState ftsTrack = simTrackToFts(simTrack, simVertex);

  TrajectoryStateOnSurface tsof = atStation2(ftsTrack, simTrack.momentum().eta());  //propagation

  return tsof;
}

TrajectoryStateOnSurface CandidateSimMuonMatcher::propagate(const TrackingParticle& trackingParticle) {
  FreeTrajectoryState ftsTrack = simTrackToFts(trackingParticle);

  TrajectoryStateOnSurface tsof = atStation2(ftsTrack, trackingParticle.momentum().eta());  //propagation

  return tsof;
}

float normal_pdf(float x, float m, float s) {
  static const float inv_sqrt_2pi = 0.3989422804014327;
  float a = (x - m) / s;

  return inv_sqrt_2pi / s * std::exp(-0.5 * a * a);
}

MatchingResult CandidateSimMuonMatcher::match(const l1t::RegionalMuonCand* muonCand,
                                              const AlgoMuonPtr& procMuon,
                                              const SimTrack& simTrack,
                                              TrajectoryStateOnSurface& tsof) {
  MatchingResult result(simTrack);

  double candGloablEta = muonCand->hwEta() * 0.010875;
  //if (std::abs(simTrack.momentum().eta() - candGloablEta) < 0.3) //has no sense for displaced muons
  {
    double candGlobalPhi = omtfConfig->calcGlobalPhi(muonCand->hwPhi(), muonCand->processor());
    candGlobalPhi = hwGmtPhiToGlobalPhi(candGlobalPhi);

    if (candGlobalPhi > M_PI)
      candGlobalPhi = candGlobalPhi - (2. * M_PI);

    result.deltaPhi = foldPhi(tsof.globalPosition().phi() - candGlobalPhi);
    result.deltaEta = tsof.globalPosition().eta() - candGloablEta;

    result.propagatedPhi = tsof.globalPosition().phi();
    result.propagatedEta = tsof.globalPosition().eta();

    double mean = 0;
    double sigma = 1;
    //if(!fillMean)
    {
      auto ptBin = deltaPhiPropCandMean->FindBin(simTrack.momentum().pt());
      mean = deltaPhiPropCandMean->GetBinContent(ptBin);
      sigma = deltaPhiPropCandStdDev->GetBinContent(ptBin);
    }
    result.matchingLikelihood = normal_pdf(result.deltaPhi, mean, sigma);  //TODO temporary solution

    result.muonCand = muonCand;
    result.procMuon = procMuon;

    double treshold = 6. * sigma;
    if (simTrack.momentum().pt() > 20)
      treshold = 7. * sigma;
    if (simTrack.momentum().pt() > 100)
      treshold = 20. * sigma;

    if (std::abs(result.deltaPhi - mean) < treshold && std::abs(result.deltaEta) < 0.3)
      result.result = MatchingResult::ResultType::matched;

    LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::match: simTrack type " << simTrack.type() << " pt "
                                  << std::setw(8) << simTrack.momentum().pt() << " eta " << std::setw(8)
                                  << simTrack.momentum().eta() << " phi " << std::setw(8) << simTrack.momentum().phi()
                                  << " propagation eta " << std::setw(8) << tsof.globalPosition().eta() << " phi "
                                  << tsof.globalPosition().phi() << "\n             muonCand pt " << std::setw(8)
                                  << muonCand->hwPt() << " candGloablEta " << std::setw(8) << candGloablEta
                                  << " candGlobalPhi " << std::setw(8) << candGlobalPhi << " hwQual "
                                  << muonCand->hwQual() << " deltaEta " << std::setw(8) << result.deltaEta
                                  << " deltaPhi " << std::setw(8) << result.deltaPhi << " sigma " << std::setw(8)
                                  << sigma << " Likelihood " << std::setw(8) << result.matchingLikelihood << " result "
                                  << (short)result.result << std::endl;
  }

  return result;
}

MatchingResult CandidateSimMuonMatcher::match(const l1t::RegionalMuonCand* muonCand,
                                              const AlgoMuonPtr& procMuon,
                                              const TrackingParticle& trackingParticle,
                                              TrajectoryStateOnSurface& tsof) {
  MatchingResult result(trackingParticle);

  double candGloablEta = muonCand->hwEta() * 0.010875;
  //if (std::abs(trackingParticle.momentum().eta() - candGloablEta) < 0.3)  //has no sense for displaced muons
  {
    double candGlobalPhi = omtfConfig->calcGlobalPhi(muonCand->hwPhi(), muonCand->processor());
    candGlobalPhi = hwGmtPhiToGlobalPhi(candGlobalPhi);

    if (candGlobalPhi > M_PI)
      candGlobalPhi = candGlobalPhi - (2. * M_PI);

    result.deltaPhi = foldPhi(tsof.globalPosition().phi() - candGlobalPhi);
    result.deltaEta = tsof.globalPosition().eta() - candGloablEta;

    result.propagatedPhi = tsof.globalPosition().phi();
    result.propagatedEta = tsof.globalPosition().eta();

    double mean = 0;
    double sigma = 1;
    //if(!fillMean)
    {
      auto ptBin = deltaPhiPropCandMean->FindBin(trackingParticle.pt());

      mean = deltaPhiPropCandMean->GetBinContent(ptBin);
      sigma = deltaPhiPropCandStdDev->GetBinContent(ptBin);
    }

    result.matchingLikelihood = normal_pdf(result.deltaPhi, mean, sigma);  //TODO temporary solution

    result.muonCand = muonCand;
    result.procMuon = procMuon;

    double treshold = 6. * sigma;
    if (trackingParticle.pt() > 20)
      treshold = 7. * sigma;
    if (trackingParticle.pt() > 100)
      treshold = 20. * sigma;

    if (std::abs(result.deltaPhi - mean) < treshold && std::abs(result.deltaEta) < 0.3)
      result.result = MatchingResult::ResultType::matched;

    LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::match: trackingParticle type "
                                  << trackingParticle.pdgId() << " pt " << std::setw(8) << trackingParticle.pt()
                                  << " eta " << std::setw(8) << trackingParticle.momentum().eta() << " phi "
                                  << std::setw(8) << trackingParticle.momentum().phi() << " propagation eta "
                                  << std::setw(8) << tsof.globalPosition().eta() << " phi "
                                  << tsof.globalPosition().phi() << " muonCand pt " << std::setw(8) << muonCand->hwPt()
                                  << " candGloablEta " << std::setw(8) << candGloablEta << " candGlobalPhi "
                                  << std::setw(8) << candGlobalPhi << " hwQual " << muonCand->hwQual() << " deltaEta "
                                  << std::setw(8) << result.deltaEta << " deltaPhi " << std::setw(8) << result.deltaPhi
                                  << " Likelihood " << std::setw(8) << result.matchingLikelihood << " result "
                                  << (short)result.result << std::endl;
  }

  return result;
}

std::vector<MatchingResult> CandidateSimMuonMatcher::cleanMatching(std::vector<MatchingResult> matchingResults,
                                                                   std::vector<const l1t::RegionalMuonCand*>& muonCands,
                                                                   AlgoMuons& ghostBustedProcMuons) {
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
    if (matchingResult.result == MatchingResult::ResultType::matched ||
        matchingResult.muonCand ==
            nullptr)  //adding also the simTracks that are not matched at all, before it is assured that they are not duplicates
      cleanedMatchingResults.push_back(matchingResult);
    if (matchingResult.result == MatchingResult::ResultType::matched) {
      /* TODO fix if filling of the deltaPhiPropCandMean and deltaPhiPropCandStdDev is needed
      if(fillMean) {
      double ptGen  = matchingResult.genPt;
        deltaPhiPropCandMean->Fill(ptGen, matchingResult.deltaPhi); //filling overflow is ok here
        deltaPhiPropCandStdDev->Fill(ptGen, matchingResult.deltaPhi * matchingResult.deltaPhi);
      }*/
    }
  }

  //adding the muonCand-s that were not matched, i.e. in order to analyze them later
  unsigned int iCand = 0;
  for (auto& muonCand : muonCands) {
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
      result.procMuon = ghostBustedProcMuons.at(iCand);
      cleanedMatchingResults.push_back(result);
    }
    iCand++;
  }

  LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::cleanMatching:" << __LINE__
                                << " CandidateSimMuonMatcher::match cleanedMatchingResults:" << std::endl;
  for (auto& result : cleanedMatchingResults) {
    if (result.trackingParticle || result.simTrack)
      LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::cleanMatching:" << __LINE__ << " simTrack type "
                                    << result.pdgId << " pt " << std::setw(8) << result.genPt << " eta " << std::setw(8)
                                    << result.genEta << " phi " << std::setw(8) << result.genPhi;
    else
      LogTrace("l1tOmtfEventPrint") << "no matched track ";

    if (result.muonCand) {
      LogTrace("l1tOmtfEventPrint") << " muonCand pt " << std::setw(8) << result.muonCand->hwPt() << " hwQual "
                                    << result.muonCand->hwQual() << " hwEta " << result.muonCand->hwEta()
                                    << " deltaEta " << std::setw(8) << result.deltaEta << " deltaPhi " << std::setw(8)
                                    << result.deltaPhi << " Likelihood " << std::setw(8) << result.matchingLikelihood
                                    << " result " << (short)result.result;
      LogTrace("l1tOmtfEventPrint") << " procMuon " << *(result.procMuon) << std::endl;
    } else
      LogTrace("l1tOmtfEventPrint") << " no muonCand "
                                    << " result " << (short)result.result << std::endl;
  }
  LogTrace("l1tOmtfEventPrint") << " " << std::endl;

  return cleanedMatchingResults;
}

std::vector<MatchingResult> CandidateSimMuonMatcher::match(std::vector<const l1t::RegionalMuonCand*>& muonCands,
                                                           AlgoMuons& ghostBustedProcMuons,
                                                           const edm::SimTrackContainer* simTracks,
                                                           const edm::SimVertexContainer* simVertices,
                                                           std::function<bool(const SimTrack&)> const& simTrackFilter) {
  std::vector<MatchingResult> matchingResults;

  for (auto& simTrack : *simTracks) {
    if (!simTrackFilter(simTrack))
      continue;

    LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::match, simTrack type " << std::setw(3) << simTrack.type()
                                  << " pt " << std::setw(9) << simTrack.momentum().pt() << " eta " << std::setw(9)
                                  << simTrack.momentum().eta() << " phi " << std::setw(9) << simTrack.momentum().phi()
                                  << std::endl;

    bool matched = false;

    TrajectoryStateOnSurface tsof = propagate(simTrack, simVertices);
    if (!tsof.isValid()) {
      LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " propagation failed" << std::endl;
      MatchingResult result;
      result.result = MatchingResult::ResultType::propagationFailed;
      continue;  //no sense to do matching
    }

    //checking if the propagated track is inside the OMTF range, TODO - tune the range!!!!!!!!!!!!!!!!!
    //eta 0.7 is the beginning of the MB2, while 1.31 is mid of RE2 + some margin
    //the eta range wider than the nominal OMTF region is needed, as in any case muons outside this region are seen by the OMTF
    //so it better to train the nn suich that is able to measure its pt, as it may affect the rate
    if ((std::abs(tsof.globalPosition().eta()) >= 0.7) && (std::abs(tsof.globalPosition().eta()) <= 1.31)) {
      LogTrace("l1tOmtfEventPrint")
          << "CandidateSimMuonMatcher::match simTrack IS in OMTF region, matching to the omtfCands";
    } else {
      LogTrace("l1tOmtfEventPrint") << "simTrack NOT in OMTF region ";
      continue;
    }

    /* TODO fix if filling of the deltaPhiPropCandMean and deltaPhiPropCandStdDev is needed
    double ptGen = simTrack.momentum().pt();
    if(ptGen >= deltaPhiVertexProp->GetXaxis()->GetXmax())
      ptGen = deltaPhiVertexProp->GetXaxis()->GetXmax() - 0.01;

    deltaPhiVertexProp->Fill(ptGen, simTrack.momentum().phi() - tsof.globalPosition().phi());*/

    unsigned int iCand = 0;
    for (auto& muonCand : muonCands) {
      //dropping very low quality candidates, as they are fakes usually - but it has no sense, then the results are not conclusive
      //if(muonCand->hwQual() > 1)
      {
        MatchingResult result;
        if (tsof.isValid()) {
          result = match(muonCand, ghostBustedProcMuons.at(iCand), simTrack, tsof);
        }
        int vtxInd = simTrack.vertIndex();
        if (vtxInd >= 0) {
          result.simVertex = &(
              simVertices->at(vtxInd));  //TODO ?????? something strange is here, was commented in the previous version
        }
        if (result.result == MatchingResult::ResultType::matched) {
          matchingResults.push_back(result);
          matched = true;
        }
      }
      iCand++;
    }

    if (!matched) {  //we are adding also if it was not matching to any candidate
      MatchingResult result(simTrack);
      matchingResults.push_back(result);
      LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " no matching candidate found" << std::endl;
    }
  }

  return cleanMatching(matchingResults, muonCands, ghostBustedProcMuons);
}

std::vector<MatchingResult> CandidateSimMuonMatcher::match(
    std::vector<const l1t::RegionalMuonCand*>& muonCands,
    AlgoMuons& ghostBustedProcMuons,
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
                                  << " eta " << std::setw(9) << trackingParticle.momentum().eta() << " phi "
                                  << std::setw(9) << trackingParticle.momentum().phi() << std::endl;

    bool matched = false;

    TrajectoryStateOnSurface tsof = propagate(trackingParticle);
    if (!tsof.isValid()) {
      LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::match:" << __LINE__ << " propagation failed"
                                    << std::endl;
      MatchingResult result;
      result.result = MatchingResult::ResultType::propagationFailed;
      continue;  //no sense to do matching
    }

    LogTrace("l1tOmtfEventPrint") << "CandidateSimMuonMatcher::match, tsof.globalPosition().eta() "
                                  << tsof.globalPosition().eta();

    //checking if the propagated track is inside the OMTF range, TODO - tune the range!!!!!!!!!!!!!!!!!
    //eta 0.7 is the beginning of the MB2, while 1.31 is mid of RE2 + some margin
    if ((std::abs(tsof.globalPosition().eta()) >= 0.7) && (std::abs(tsof.globalPosition().eta()) <= 1.31)) {
      LogTrace("l1tOmtfEventPrint")
          << "CandidateSimMuonMatcher::match trackingParticle IS in OMTF region, matching to the omtfCands";
    } else {
      LogTrace("l1tOmtfEventPrint") << "trackingParticle NOT in OMTF region ";
      continue;
    }

    /* TODO fix if filling of the deltaPhiPropCandMean and deltaPhiPropCandStdDev is needed
    double ptGen = trackingParticle.pt();
    if(ptGen >= deltaPhiVertexProp->GetXaxis()->GetXmax())
      ptGen = deltaPhiVertexProp->GetXaxis()->GetXmax() - 0.01;

    deltaPhiVertexProp->Fill(ptGen, trackingParticle.momentum().phi() - tsof.globalPosition().phi());
*/

    unsigned int iCand = 0;
    for (auto& muonCand : muonCands) {
      //dropping very low quality candidates, as they are fakes usually - but it has no sense, then the results are not conclusive then
      /*if(muonCand->hwQual() <= 1)
        continue; */

      MatchingResult result;
      if (tsof.isValid()) {
        result = match(muonCand, ghostBustedProcMuons.at(iCand), trackingParticle, tsof);
      }
      iCand++;

      if (result.result == MatchingResult::ResultType::matched) {
        matchingResults.push_back(result);
        matched = true;
      }
    }

    if (!matched) {  //we are adding result also if it there was no matching to any candidate
      MatchingResult result(trackingParticle);
      matchingResults.push_back(result);
      LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " no matching candidate found" << std::endl;
    }
  }

  return cleanMatching(matchingResults, muonCands, ghostBustedProcMuons);
}
