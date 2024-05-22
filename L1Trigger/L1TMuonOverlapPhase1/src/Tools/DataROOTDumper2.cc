/*
 * DataROOTDumper2.cc
 *
 *  Created on: Dec 11, 2019
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/DataROOTDumper2.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "TFile.h"
#include "TTree.h"

DataROOTDumper2::DataROOTDumper2(const edm::ParameterSet& edmCfg,
                                 const OMTFConfiguration* omtfConfig,
                                 CandidateSimMuonMatcher* candidateSimMuonMatcher)
    : EmulationObserverBase(edmCfg, omtfConfig), candidateSimMuonMatcher(candidateSimMuonMatcher) {
  edm::LogVerbatim("l1tOmtfEventPrint") << " omtfConfig->nTestRefHits() " << omtfConfig->nTestRefHits()
                                        << " event.omtfGpResultsPdfSum.num_elements() " << endl;
  initializeTTree();

  if (edmCfg.exists("dumpKilledOmtfCands"))
    if (edmCfg.getParameter<bool>("dumpKilledOmtfCands"))
      dumpKilledOmtfCands = true;

  if (edmCfg.exists("candidateSimMuonMatcherType")) {
    if (edmCfg.getParameter<std::string>("candidateSimMuonMatcherType") == "propagation")
      usePropagation = true;
    else if (edmCfg.getParameter<std::string>("candidateSimMuonMatcherType") == "matchSimple")
      usePropagation = false;

    edm::LogImportant("l1tOmtfEventPrint")
        << " CandidateSimMuonMatcher: candidateSimMuonMatcherType "
        << edmCfg.getParameter<std::string>("candidateSimMuonMatcherType") << std::endl;
  }

  edm::LogVerbatim("l1tOmtfEventPrint") << " DataROOTDumper2 created. dumpKilledOmtfCands " << dumpKilledOmtfCands
                                        << std::endl;
}

DataROOTDumper2::~DataROOTDumper2() {}

void DataROOTDumper2::initializeTTree() {
  edm::Service<TFileService> fs;

  rootTree = fs->make<TTree>("OMTFHitsTree", "");

  rootTree->Branch("eventNum", &omtfEvent.eventNum);
  rootTree->Branch("muonEvent", &omtfEvent.muonEvent);

  rootTree->Branch("muonPt", &omtfEvent.muonPt);
  rootTree->Branch("muonEta", &omtfEvent.muonEta);
  rootTree->Branch("muonPhi", &omtfEvent.muonPhi);
  rootTree->Branch("muonPropEta", &omtfEvent.muonPropEta);
  rootTree->Branch("muonPropPhi", &omtfEvent.muonPropPhi);
  rootTree->Branch("muonCharge", &omtfEvent.muonCharge);

  rootTree->Branch("muonDxy", &omtfEvent.muonDxy);
  rootTree->Branch("muonRho", &omtfEvent.muonRho);

  rootTree->Branch("omtfPt", &omtfEvent.omtfPt);
  rootTree->Branch("omtfUPt", &omtfEvent.omtfUPt);
  rootTree->Branch("omtfEta", &omtfEvent.omtfEta);
  rootTree->Branch("omtfPhi", &omtfEvent.omtfPhi);
  rootTree->Branch("omtfCharge", &omtfEvent.omtfCharge);

  rootTree->Branch("omtfHwEta", &omtfEvent.omtfHwEta);

  rootTree->Branch("omtfProcessor", &omtfEvent.omtfProcessor);
  rootTree->Branch("omtfScore", &omtfEvent.omtfScore);
  rootTree->Branch("omtfQuality", &omtfEvent.omtfQuality);
  rootTree->Branch("omtfRefLayer", &omtfEvent.omtfRefLayer);
  rootTree->Branch("omtfRefHitNum", &omtfEvent.omtfRefHitNum);

  rootTree->Branch("omtfFiredLayers", &omtfEvent.omtfFiredLayers);  //<<<<<<<<<<<<<<<<<<<<<<!!!!TODOO

  rootTree->Branch("killed", &omtfEvent.killed);

  rootTree->Branch("hits", &omtfEvent.hits);

  rootTree->Branch("deltaEta", &omtfEvent.deltaEta);
  rootTree->Branch("deltaPhi", &omtfEvent.deltaPhi);

  ptGenPos = fs->make<TH1I>("ptGenPos", "ptGenPos, eta at vertex 0.8 - 1.24", 400, 0, 200);  //TODO
  ptGenNeg = fs->make<TH1I>("ptGenNeg", "ptGenNeg, eta at vertex 0.8 - 1.24", 400, 0, 200);
}

void DataROOTDumper2::observeProcesorEmulation(unsigned int iProcessor,
                                               l1t::tftype mtfType,
                                               const std::shared_ptr<OMTFinput>&,
                                               const AlgoMuons& algoCandidates,
                                               const AlgoMuons& gbCandidates,
                                               const std::vector<l1t::RegionalMuonCand>& candMuons) {}

void DataROOTDumper2::observeEventEnd(const edm::Event& iEvent,
                                      std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) {
  /*
  int muonCharge = 0;
  if (simMuon) {
    if (std::abs(simMuon->momentum().eta()) < 0.8 || std::abs(simMuon->momentum().eta()) > 1.24)
      return;

    muonCharge = (std::abs(simMuon->type()) == 13) ? simMuon->type() / -13 : 0;
    if (muonCharge > 0)
      ptGenPos->Fill(simMuon->momentum().pt());
    else
      ptGenNeg->Fill(simMuon->momentum().pt());
  }

  if (simMuon == nullptr || !omtfCand->isValid())  //no sim muon or empty candidate
    return;

  omtfEvent.muonPt = simMuon->momentum().pt();
  omtfEvent.muonEta = simMuon->momentum().eta();

  //TODO add cut on ete if needed
    if(std::abs(event.muonEta) < 0.8 || std::abs(event.muonEta) > 1.24)
    return;

  omtfEvent.muonPhi = simMuon->momentum().phi();
  omtfEvent.muonCharge = muonCharge;  //TODO
   */

  std::vector<MatchingResult> matchingResults = candidateSimMuonMatcher->getMatchingResults();
  LogTrace("l1tOmtfEventPrint") << "\nDataROOTDumper2::observeEventEnd matchingResults.size() "
                                << matchingResults.size() << std::endl;

  //candidateSimMuonMatcher should use the  trackingParticles, because the simTracks are not stored for the pile-up events

  //for some events there are more than one matchingResults,
  //Usually at least one them has  genPt 0, which means no simMuon was matched, so candidate is ghost (or fake)
  //so better is to to drop such event, as it is not sue if the correct simMuon was matched to the candidate.
  //So we assume here that when the propagation is not used it is a single mu sample and this filter has sense
  //the propagation is used for multi-muon sample, so then this fitler cannot be used
  //TODO add a flag to enable this filter? Disable it if not needed
  if (!usePropagation && matchingResults.size() > 1) {  //omtfConfig->cleanStubs() &&
    edm::LogVerbatim("l1tOmtfEventPrint")
        << "\nDataROOTDumper2::observeEventEnd matchingResults.size() " << matchingResults.size() << std::endl;

    for (auto& matchingResult : matchingResults) {
      edm::LogVerbatim("l1tOmtfEventPrint") << "matchingResult: genPt " << matchingResult.genPt;
      if (matchingResult.procMuon)
        edm::LogVerbatim("l1tOmtfEventPrint") << " procMuon.PtConstr " << matchingResult.procMuon->getPtConstr();
      else
        edm::LogVerbatim("l1tOmtfEventPrint") << " no procMuon" << std::endl;
    }
    edm::LogVerbatim("l1tOmtfEventPrint") << "dropping the event!!!\n" << std::endl;
    return;
  }

  for (auto& matchingResult : matchingResults) {
    omtfEvent.eventNum = iEvent.id().event();

    if (matchingResult.trackingParticle) {
      auto trackingParticle = matchingResult.trackingParticle;

      omtfEvent.muonEvent = trackingParticle->eventId().event();

      omtfEvent.muonPt = trackingParticle->pt();
      omtfEvent.muonEta = trackingParticle->momentum().eta();
      omtfEvent.muonPhi = trackingParticle->momentum().phi();
      omtfEvent.muonPropEta = matchingResult.propagatedEta;
      omtfEvent.muonPropPhi = matchingResult.propagatedPhi;
      omtfEvent.muonCharge = (std::abs(trackingParticle->pdgId()) == 13) ? trackingParticle->pdgId() / -13 : 0;

      if (trackingParticle->parentVertex().isNonnull()) {
        omtfEvent.muonDxy = trackingParticle->dxy();
        omtfEvent.muonRho = trackingParticle->parentVertex()->position().Rho();
      }

      omtfEvent.deltaEta = matchingResult.deltaEta;
      omtfEvent.deltaPhi = matchingResult.deltaPhi;

      LogTrace("l1tOmtfEventPrint") << "DataROOTDumper2::observeEventEnd trackingParticle: eventId "
                                    << trackingParticle->eventId().event() << " pdgId " << std::setw(3)
                                    << trackingParticle->pdgId() << " trackId "
                                    << trackingParticle->g4Tracks().at(0).trackId() << " pt " << std::setw(9)
                                    << trackingParticle->pt()  //<<" Beta "<<simMuon->momentum().Beta()
                                    << " eta " << std::setw(9) << trackingParticle->momentum().eta() << " phi "
                                    << std::setw(9) << trackingParticle->momentum().phi() << std::endl;

      if (std::abs(omtfEvent.muonEta) > 0.8 && std::abs(omtfEvent.muonEta) < 1.24) {
        if (omtfEvent.muonCharge > 0)
          ptGenPos->Fill(omtfEvent.muonPt);
        else
          ptGenNeg->Fill(omtfEvent.muonPt);
      }
    } else if (matchingResult.simTrack) {
      auto simTrack = matchingResult.simTrack;

      omtfEvent.muonEvent = simTrack->eventId().event();

      omtfEvent.muonPt = simTrack->momentum().pt();
      omtfEvent.muonEta = simTrack->momentum().eta();
      omtfEvent.muonPhi = simTrack->momentum().phi();
      omtfEvent.muonPropEta = matchingResult.propagatedEta;
      omtfEvent.muonPropPhi = matchingResult.propagatedPhi;
      omtfEvent.muonCharge = simTrack->charge();

      if (!simTrack->noVertex() && matchingResult.simVertex) {
        const math::XYZTLorentzVectorD& vtxPos = matchingResult.simVertex->position();
        omtfEvent.muonDxy = (-vtxPos.X() * simTrack->momentum().py() + vtxPos.Y() * simTrack->momentum().px()) /
                            simTrack->momentum().pt();
        omtfEvent.muonRho = vtxPos.Rho();
      }

      omtfEvent.deltaEta = matchingResult.deltaEta;
      omtfEvent.deltaPhi = matchingResult.deltaPhi;

      LogTrace("l1tOmtfEventPrint") << "DataROOTDumper2::observeEventEnd simTrack: eventId "
                                    << simTrack->eventId().event() << " pdgId " << std::setw(3)
                                    << simTrack->type()  //<< " trackId " << simTrack->g4Tracks().at(0).trackId()
                                    << " pt " << std::setw(9)
                                    << simTrack->momentum().pt()  //<<" Beta "<<simMuon->momentum().Beta()
                                    << " eta " << std::setw(9) << simTrack->momentum().eta() << " phi " << std::setw(9)
                                    << simTrack->momentum().phi() << std::endl;

      if (std::abs(omtfEvent.muonEta) > 0.8 && std::abs(omtfEvent.muonEta) < 1.24) {
        if (omtfEvent.muonCharge > 0)
          ptGenPos->Fill(omtfEvent.muonPt);
        else
          ptGenNeg->Fill(omtfEvent.muonPt);
      }
    } else {
      omtfEvent.muonEvent = -1;

      omtfEvent.muonPt = 0;

      omtfEvent.muonEta = 0;
      omtfEvent.muonPhi = 0;

      omtfEvent.muonPropEta = 0;
      omtfEvent.muonPropPhi = 0;

      omtfEvent.muonCharge = 0;  //TODO

      omtfEvent.muonDxy = 0;
      omtfEvent.muonRho = 0;
    }

    auto addOmtfCand = [&](AlgoMuonPtr& procMuon) {
      omtfEvent.omtfPt = omtfConfig->hwPtToGev(procMuon->getPtConstr());
      omtfEvent.omtfUPt = omtfConfig->hwUPtToGev(procMuon->getPtUnconstr());
      omtfEvent.omtfEta = omtfConfig->hwEtaToEta(procMuon->getEtaHw());
      omtfEvent.omtfPhi = procMuon->getPhi();
      omtfEvent.omtfCharge = procMuon->getChargeConstr();
      omtfEvent.omtfScore = procMuon->getPdfSum();

      omtfEvent.omtfHwEta = procMuon->getEtaHw();

      omtfEvent.omtfFiredLayers = procMuon->getFiredLayerBits();
      omtfEvent.omtfRefLayer = procMuon->getRefLayer();
      omtfEvent.omtfRefHitNum = procMuon->getRefHitNumber();

      omtfEvent.hits.clear();

      //TODO choose, which gpResult should be dumped
      //auto& gpResult = procMuon->getGpResultConstr();
      auto& gpResult = (procMuon->getGpResultUnconstr().getPdfSumUnconstr() > procMuon->getGpResultConstr().getPdfSum())
                           ? procMuon->getGpResultUnconstr()
                           : procMuon->getGpResultConstr();

      /*
        edm::LogVerbatim("l1tOmtfEventPrint")<<"DataROOTDumper2:;observeEventEnd muonPt "<<event.muonPt<<" muonCharge "<<event.muonCharge
            <<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer<<" omtfPtCont "<<event.omtfPtCont
            <<std::endl;  */

      for (unsigned int iLogicLayer = 0; iLogicLayer < gpResult.getStubResults().size(); ++iLogicLayer) {
        auto& stubResult = gpResult.getStubResults()[iLogicLayer];

        //TODO it is to have the hit if it is below the quality cut
        /*if (omtfConfig->isBendingLayer(iLogicLayer) && !stubResult.getMuonStub()) {
          auto& stubResult = gpResult.getStubResults()[iLogicLayer-1];
        }*/

        if (stubResult.getMuonStub()) {  //&& stubResult.getValid() //TODO!!!!!!!!!!!!!!!!1
          OmtfEvent::Hit hit;
          hit.layer = iLogicLayer;
          hit.quality = stubResult.getMuonStub()->qualityHw;
          hit.eta = stubResult.getMuonStub()->etaHw;  //in which scale?
          hit.valid = stubResult.getValid();

          int hitPhi = stubResult.getMuonStub()->phiHw;
          unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[procMuon->getRefLayer()];
          int phiRefHit = gpResult.getStubResults()[refLayerLogicNum].getMuonStub()->phiHw;

          if (omtfConfig->isBendingLayer(iLogicLayer)) {
            hitPhi = stubResult.getMuonStub()->phiBHw;
            phiRefHit = 0;  //phi ref hit for the bending layer set to 0, since it should not be included in the phiDist
          }

          //phiDist = hitPhi - phiRefHit;
          hit.phiDist = hitPhi - phiRefHit;

          /* LogTrace("l1tOmtfEventPrint")<<" muonPt "<<event.muonPt<<" omtfPt "<<event.omtfPt<<" RefLayer "<<event.omtfRefLayer
                <<" layer "<<int(hit.layer)<<" PdfBin "<<stubResult.getPdfBin()<<" hit.phiDist "<<hit.phiDist<<" valid "<<stubResult.getValid()<<" " //<<" phiDist "<<phiDist
                <<" getDistPhiBitShift "<<procMuon->getGoldenPatern()->getDistPhiBitShift(iLogicLayer, procMuon->getRefLayer())
                <<" meanDistPhiValue   "<<procMuon->getGoldenPatern()->meanDistPhiValue(iLogicLayer, procMuon->getRefLayer())//<<(phiDist != hit.phiDist? "!!!!!!!<<<<<" : "")
                <<endl;*/

          /*if (hit.phiDist > 504 || hit.phiDist < -512) {
            edm::LogVerbatim("l1tOmtfEventPrint")
                << " muonPt " << omtfEvent.muonPt << " omtfPt " << omtfEvent.omtfPt << " RefLayer "
                << (int)omtfEvent.omtfRefLayer << " layer " << int(hit.layer) << " hit.phiDist " << hit.phiDist
                << " valid " << stubResult.getValid() << " !!!!!!!!!!!!!!!!!!!!!!!!" << endl;
          }*/

          DetId detId(stubResult.getMuonStub()->detId);
          if (detId.subdetId() == MuonSubdetId::CSC) {
            CSCDetId cscId(detId);
            hit.z = cscId.chamber() % 2;
          }

          omtfEvent.hits.push_back(hit.rawData);
        }
      }

      LogTrace("l1tOmtfEventPrint") << "DataROOTDumper2::observeEventEnd adding omtfCand : " << std::endl;
      auto finalCandidate = matchingResult.muonCand;
      LogTrace("l1tOmtfEventPrint") << " hwPt " << finalCandidate->hwPt() << " hwSign " << finalCandidate->hwSign()
                                    << " hwQual " << finalCandidate->hwQual() << " hwEta " << std::setw(4)
                                    << finalCandidate->hwEta() << std::setw(4) << " hwPhi " << finalCandidate->hwPhi()
                                    << "    eta " << std::setw(9) << (finalCandidate->hwEta() * 0.010875)
                                    << " isKilled " << procMuon->isKilled() << " tRefLayer " << procMuon->getRefLayer()
                                    << " RefHitNumber " << procMuon->getRefHitNumber() << std::endl;
    };

    if (matchingResult.muonCand && matchingResult.procMuon->getPtConstr() > 0 &&
        matchingResult.muonCand->hwQual() >= 1) {
      //TODO set the quality, quality 0 has the candidates with eta > 1.3(?) EtaHw >= 121
      //&& matchingResult.genPt < 20

      omtfEvent.omtfQuality = matchingResult.muonCand->hwQual();  //procMuon->getQ();
      omtfEvent.killed = false;
      omtfEvent.omtfProcessor = matchingResult.muonCand->processor();

      if (matchingResult.muonCand->trackFinderType() == l1t::omtf_neg) {
        omtfEvent.omtfProcessor *= -1;
      }

      addOmtfCand(matchingResult.procMuon);
      rootTree->Fill();

      if (dumpKilledOmtfCands) {
        for (auto& killedCand : matchingResult.procMuon->getKilledMuons()) {
          omtfEvent.omtfQuality = 0;
          omtfEvent.killed = true;
          if (killedCand->isKilled() == false) {
            edm::LogVerbatim("l1tOmtfEventPrint") << " killedCand->isKilled() == false !!!!!!!!";
          }
          addOmtfCand(killedCand);
          rootTree->Fill();
        }
      }
    } else if (omtfEvent.muonPt > 0) {  //checking if there was a simMuon
      LogTrace("l1tOmtfEventPrint") << "DataROOTDumper2::observeEventEnd no matching omtfCand" << std::endl;

      omtfEvent.omtfPt = 0;
      omtfEvent.omtfUPt = 0;
      omtfEvent.omtfEta = 0;
      omtfEvent.omtfPhi = 0;
      omtfEvent.omtfCharge = 0;
      omtfEvent.omtfScore = 0;

      omtfEvent.omtfHwEta = 0;

      omtfEvent.omtfFiredLayers = 0;
      omtfEvent.omtfRefLayer = 0;
      omtfEvent.omtfRefHitNum = 0;
      omtfEvent.omtfProcessor = 10;

      omtfEvent.omtfQuality = 0;
      omtfEvent.killed = false;

      omtfEvent.hits.clear();

      rootTree->Fill();
    }
  }
  evntCnt++;
}

void DataROOTDumper2::endJob() { edm::LogVerbatim("l1tOmtfEventPrint") << " evntCnt " << evntCnt << endl; }
