/*
 * EventCapture.cpp
 *
 *  Created on: Oct 23, 2019
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/EventCapture.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OmtfName.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinputMaker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <memory>
#include <sstream>

EventCapture::EventCapture(const edm::ParameterSet& edmCfg,
                           const OMTFConfiguration* omtfConfig,
                           CandidateSimMuonMatcher* candidateSimMuonMatcher,
                           const MuonGeometryTokens& muonGeometryTokens,
                           const GoldenPatternVec<GoldenPattern>* gps)
    : omtfConfig(omtfConfig),
      goldenPatterns(gps),
      candidateSimMuonMatcher(candidateSimMuonMatcher),
      inputInProcs(omtfConfig->processorCnt()),
      algoMuonsInProcs(omtfConfig->processorCnt()),
      gbCandidatesInProcs(omtfConfig->processorCnt()) {
  //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__<<" omtfConfig->nProcessors() "<<omtfConfig->nProcessors()<<std::endl;
  if (edmCfg.exists("simTracksTag"))
    simTracksTag = edmCfg.getParameter<edm::InputTag>("simTracksTag");
  else
    edm::LogImportant("OMTFReconstruction")
        << "EventCapture::EventCapture: no InputTag simTracksTag found" << std::endl;

  if (this->candidateSimMuonMatcher)
    stubsSimHitsMatcher = std::make_unique<StubsSimHitsMatcher>(edmCfg, omtfConfig, muonGeometryTokens);
}

EventCapture::~EventCapture() {
  // TODO Auto-generated destructor stub
}

void EventCapture::beginRun(edm::EventSetup const& eventSetup) {
  if (stubsSimHitsMatcher)
    stubsSimHitsMatcher->beginRun(eventSetup);
}

void EventCapture::observeEventBegin(const edm::Event& event) {
  simMuons.clear();

  if (!simTracksTag.label().empty()) {
    edm::Handle<edm::SimTrackContainer> simTraksHandle;
    event.getByLabel(simTracksTag, simTraksHandle);

    for (unsigned int iSimTrack = 0; iSimTrack != simTraksHandle->size(); iSimTrack++) {
      if (abs((*simTraksHandle.product())[iSimTrack].type()) == 13)
        simMuons.emplace_back(simTraksHandle, iSimTrack);
    }
  }

  for (auto& input : inputInProcs)
    input.reset();

  for (auto& algoMuonsInProc : algoMuonsInProcs)
    algoMuonsInProc.clear();

  for (auto& gbCandidatesInProc : gbCandidatesInProcs)
    gbCandidatesInProc.clear();
}

void EventCapture::observeProcesorEmulation(unsigned int iProcessor,
                                            l1t::tftype mtfType,
                                            const std::shared_ptr<OMTFinput>& input,
                                            const AlgoMuons& algoCandidates,
                                            const AlgoMuons& gbCandidates,
                                            const std::vector<l1t::RegionalMuonCand>& candMuons) {
  unsigned int procIndx = omtfConfig->getProcIndx(iProcessor, mtfType);

  inputInProcs[procIndx] = input;

  algoMuonsInProcs[procIndx] = algoCandidates;
  gbCandidatesInProcs[procIndx] = gbCandidates;
}

void EventCapture::observeEventEnd(const edm::Event& iEvent,
                                   std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) {
  std::ostringstream ostr;
  //filtering

  bool dump = false;

  if (candidateSimMuonMatcher) {
    std::vector<MatchingResult> matchingResults = candidateSimMuonMatcher->getMatchingResults();
    edm::LogVerbatim("l1tOmtfEventPrint") << "matchingResults.size() " << matchingResults.size() << std::endl;

    //candidateSimMuonMatcher should use the  trackingParticles, because the simTracks are not stored for the pile-up events
    for (auto& matchingResult : matchingResults) {
      if (matchingResult.muonCand && matchingResult.muonCand->hwQual() >= 12 &&
          matchingResult.muonCand->hwPt() > 38) {  //&& matchingResult.genPt < 20
        dump = true;

        bool runStubsSimHitsMatcher = false;
        if (matchingResult.trackingParticle) {
          auto trackingParticle = matchingResult.trackingParticle;
          ostr << "trackingParticle: eventId " << trackingParticle->eventId().event() << " pdgId " << std::setw(3)
               << trackingParticle->pdgId() << " trackId " << trackingParticle->g4Tracks().at(0).trackId() << " pt "
               << std::setw(9) << trackingParticle->pt()  //<<" Beta "<<simMuon->momentum().Beta()
               << " eta " << std::setw(9) << trackingParticle->momentum().eta() << " phi " << std::setw(9)
               << trackingParticle->momentum().phi() << std::endl;
        } else {
          ostr << "no simMuon ";
          runStubsSimHitsMatcher = true;
        }
        ostr << "matched to: " << std::endl;
        auto finalCandidate = matchingResult.muonCand;
        ostr << " hwPt " << finalCandidate->hwPt() << " hwSign " << finalCandidate->hwSign() << " hwQual "
             << finalCandidate->hwQual() << " hwEta " << std::setw(4) << finalCandidate->hwEta() << std::setw(4)
             << " hwPhi " << finalCandidate->hwPhi() << "    eta " << std::setw(9)
             << (finalCandidate->hwEta() * 0.010875) << " phi " << std::endl;

        if (runStubsSimHitsMatcher)
          stubsSimHitsMatcher->match(iEvent, matchingResult.muonCand, matchingResult.procMuon, ostr);
      }
    }
  } else if (!simTracksTag.label().empty()) {
    dump = false;
    bool wasSimMuInOmtfPos = false;
    bool wasSimMuInOmtfNeg = false;
    for (auto& simMuon : simMuons) {
      if (simMuon->eventId().event() == 0 && abs(simMuon->momentum().eta()) > 0.82 &&
          abs(simMuon->momentum().eta()) < 1.24 && simMuon->momentum().pt() >= 3.) {
        ostr << "SimMuon: eventId " << simMuon->eventId().event() << " pdgId " << std::setw(3) << simMuon->type()
             << " pt " << std::setw(9) << simMuon->momentum().pt()  //<<" Beta "<<simMuon->momentum().Beta()
             << " eta " << std::setw(9) << simMuon->momentum().eta() << " phi " << std::setw(9)
             << simMuon->momentum().phi() << std::endl;

        if (simMuon->momentum().eta() > 0)
          wasSimMuInOmtfPos = true;
        else
          wasSimMuInOmtfNeg = true;
      }
    }

    bool wasCandInNeg = false;
    bool wasCandInPos = false;

    for (auto& finalCandidate : *finalCandidates) {
      if (finalCandidate.trackFinderType() == l1t::tftype::omtf_neg && finalCandidate.hwQual() >= 12 &&
          finalCandidate.hwPt() > 20)
        wasCandInNeg = true;

      if (finalCandidate.trackFinderType() == l1t::tftype::omtf_pos && finalCandidate.hwQual() >= 12 &&
          finalCandidate.hwPt() > 20)
        wasCandInPos = true;
    }

    if ((wasSimMuInOmtfNeg && wasCandInNeg))  //TODO
      dump = true;

    if ((wasSimMuInOmtfPos && wasCandInPos))  //TODO
      dump = true;
  }

  /*  dump = true; ///TODO if present then dumps all events!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if(!dump)
    return;

  bool dump = false;
  for (auto& finalCandidate : *finalCandidates) {
    if (finalCandidate.hwPt() < 41) {  //  finalCandidate.hwQual() >= 1  41
      dump = true;
    }
  }*/

  dump = true;  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  //!!!!!!!!!!!!!!!!!!!!!!!! TODO if present then dumps all events!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  if (!dump)
    return;

  ///printing

  edm::LogVerbatim("l1tOmtfEventPrint") << "##################### EventCapture::observeEventEnd - dump of event "
                                        << iEvent.id() << " #####################################################"
                                        << std::endl;

  edm::LogVerbatim("l1tOmtfEventPrint") << ostr.str() << endl;  //printing sim muons

  edm::LogVerbatim("l1tOmtfEventPrint") << "finalCandidates " << std::endl;
  for (int bx = finalCandidates->getFirstBX(); bx <= finalCandidates->getLastBX(); bx++) {
    for (auto finalCandidateIt = finalCandidates->begin(bx); finalCandidateIt != finalCandidates->end(bx);
         finalCandidateIt++) {
      auto& finalCandidate = *finalCandidateIt;
      int globHwPhi = (finalCandidate.processor()) * 96 + finalCandidate.hwPhi();
      // first processor starts at CMS phi = 15 degrees (24 in int)... Handle wrap-around with %. Add 576 to make sure the number is positive
      globHwPhi = (globHwPhi + 600) % 576;

      double globalPhi = globHwPhi * 2. * M_PI / 576;
      if (globalPhi > M_PI)
        globalPhi = globalPhi - (2. * M_PI);

      int layerHits = (int)finalCandidate.trackAddress().at(0);
      std::bitset<18> layerHitBits(layerHits);

      edm::LogVerbatim("l1tOmtfEventPrint")
          << " bx " << bx << " hwPt " << finalCandidate.hwPt() << " hwSign " << finalCandidate.hwSign() << " hwQual "
          << finalCandidate.hwQual() << " hwEta " << std::setw(4) << finalCandidate.hwEta() << std::setw(4) << " hwPhi "
          << finalCandidate.hwPhi() << "    eta " << std::setw(9) << (finalCandidate.hwEta() * 0.010875) << " phi "
          << std::setw(9) << globalPhi << " " << layerHitBits << " processor "
          << OmtfName(finalCandidate.processor(), finalCandidate.trackFinderType()) << std::endl;

      for (auto& trackAddr : finalCandidate.trackAddress()) {
        if (trackAddr.first >= 10)
          edm::LogVerbatim("l1tOmtfEventPrint")
              << "trackAddr first " << trackAddr.first << " second " << trackAddr.second << " ptGeV "
              << omtfConfig->hwPtToGev(trackAddr.second);
      }
    }
  }
  edm::LogVerbatim("l1tOmtfEventPrint") << std::endl;

  for (unsigned int iProc = 0; iProc < inputInProcs.size(); iProc++) {
    OmtfName board(iProc);

    std::ostringstream ostrInput;
    if (inputInProcs[iProc]) {
      auto& omtfInput = *inputInProcs[iProc];
      int layersWithStubs = 0;
      for (auto& layer : omtfInput.getMuonStubs()) {
        for (auto& stub : layer) {
          bool layerFired = false;
          if (stub && (stub->type != MuonStub::Type::EMPTY)) {
            layerFired = true;

            auto globalPhiRad = omtfConfig->procHwPhiToGlobalPhi(
                stub->phiHw, OMTFinputMaker::getProcessorPhiZero(omtfConfig, iProc % 6));
            ostrInput << (*stub) << " globalPhiRad " << globalPhiRad << std::endl;
          }
          if (layerFired)
            layersWithStubs++;
        }
      }

      if (layersWithStubs != 0) {
        edm::LogVerbatim("l1tOmtfEventPrint") << "\niProcessor " << iProc << " " << board.name()
                                              << " **************************************************" << std::endl;
        edm::LogVerbatim("l1tOmtfEventPrint") << ostrInput.str() << std::endl;
      }

      if (layersWithStubs < 2)
        continue;

      edm::LogVerbatim("l1tOmtfEventPrint") << *inputInProcs[iProc] << std::endl;

      edm::LogVerbatim("l1tOmtfEventPrint") << "algoMuons " << std::endl;
      //unsigned int procIndx = omtfConfig->getProcIndx(iProcessor, mtfType);
      for (auto& algoMuon : algoMuonsInProcs[iProc]) {
        if (algoMuon->isValid()) {
          edm::LogVerbatim("l1tOmtfEventPrint")
              << board.name() << " " << *algoMuon << " RefHitNum " << algoMuon->getRefHitNumber() << std::endl;
          edm::LogVerbatim("l1tOmtfEventPrint") << algoMuon->getGpResult() << std::endl;

          if (goldenPatterns)  //watch out with the golden patterns
            for (auto& gp : *goldenPatterns) {
              if (gp->key().thePt == 0)
                continue;

              //printing GoldenPatternResult, uncomment if needed
              /*auto& gpResult = gp->getResults()[iProc][algoMuon->getRefHitNumber()];
            edm::LogVerbatim("l1tOmtfEventPrint") << " "<<gp->key() << "  "
              //<< "  refLayer: " << gpResult.getRefLayer() << "\t"
              << " Sum over layers: " << gpResult.getPdfSum() << "\t"
              << " Number of hits: " << gpResult.getFiredLayerCnt() << "\t"
              << std::endl;*/
            }
          edm::LogVerbatim("l1tOmtfEventPrint") << std::endl << std::endl;
        }
      }

      edm::LogVerbatim("l1tOmtfEventPrint") << "gbCandidates " << std::endl;
      for (auto& gbCandidate : gbCandidatesInProcs[iProc])
        if (gbCandidate->isValid())
          edm::LogVerbatim("l1tOmtfEventPrint") << board.name() << " " << *gbCandidate << std::endl;

      edm::LogVerbatim("l1tOmtfEventPrint") << std::endl;
    }
  }

  edm::LogVerbatim("l1tOmtfEventPrint") << std::endl;
}

void EventCapture::endJob() {
  if (stubsSimHitsMatcher)
    stubsSimHitsMatcher->endJob();
}
