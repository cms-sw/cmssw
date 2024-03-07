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
  if (edmCfg.exists("simTracksTag"))
    simTracksTag = edmCfg.getParameter<edm::InputTag>("simTracksTag");
  else
    edm::LogImportant("OMTFReconstruction")
        << "EventCapture::EventCapture: no InputTag simTracksTag found" << std::endl;

  //stubsSimHitsMatcher works only with the trackingParticle, because only them are stored in the pilup events
  if (this->candidateSimMuonMatcher && edmCfg.exists("trackingParticleTag"))
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
      if (std::abs((*simTraksHandle.product())[iSimTrack].type()) == 13)
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
    LogTrace("l1tOmtfEventPrint") << "matchingResults.size() " << matchingResults.size() << std::endl;

    //candidateSimMuonMatcher should use the  trackingParticles, because the simTracks are not stored for the pile-up events
    for (auto& matchingResult : matchingResults) {
      //TODO choose a condition, to print the desired candidates
      if (matchingResult.muonCand) {
        dump = true;

        bool runStubsSimHitsMatcher = false;
        if (matchingResult.trackingParticle) {
          auto trackingParticle = matchingResult.trackingParticle;
          ostr << "trackingParticle: eventId " << trackingParticle->eventId().event() << " pdgId " << std::setw(3)
               << trackingParticle->pdgId() << " trackId " << trackingParticle->g4Tracks().at(0).trackId() << " pt "
               << std::setw(9) << trackingParticle->pt()  //<<" Beta "<<simMuon->momentum().Beta()
               << " eta " << std::setw(9) << trackingParticle->momentum().eta() << " phi " << std::setw(9)
               << trackingParticle->momentum().phi() << std::endl;
        } else if (matchingResult.simTrack) {
          runStubsSimHitsMatcher = true;
          ostr << "SimMuon: eventId " << matchingResult.simTrack->eventId().event() << " pdgId " << std::setw(3)
               << matchingResult.simTrack->type() << " pt " << std::setw(9)
               << matchingResult.simTrack->momentum().pt()  //<<" Beta "<<simMuon->momentum().Beta()
               << " eta " << std::setw(9) << matchingResult.simTrack->momentum().eta() << " phi " << std::setw(9)
               << matchingResult.simTrack->momentum().phi() << std::endl;
        } else {
          ostr << "no simMuon ";
          runStubsSimHitsMatcher = true;
        }
        ostr << "matched to: " << std::endl;
        auto finalCandidate = matchingResult.muonCand;
        ostr << " hwPt " << finalCandidate->hwPt() << " hwUPt " << finalCandidate->hwPtUnconstrained() << " hwSign "
             << finalCandidate->hwSign() << " hwQual " << finalCandidate->hwQual() << " hwEta " << std::setw(4)
             << finalCandidate->hwEta() << std::setw(4) << " hwPhi " << finalCandidate->hwPhi() << "    eta "
             << std::setw(9) << (finalCandidate->hwEta() * 0.010875) << " phi " << std::endl;

        if (stubsSimHitsMatcher && runStubsSimHitsMatcher)
          stubsSimHitsMatcher->match(iEvent, matchingResult.muonCand, matchingResult.procMuon, ostr);
      }
    }
  } else if (!simTracksTag.label().empty()) {
    dump = false;
    bool wasSimMuInOmtfPos = false;
    bool wasSimMuInOmtfNeg = false;
    for (auto& simMuon : simMuons) {
      //TODO choose a condition, to print the desired events
      if (simMuon->eventId().event() == 0 && std::abs(simMuon->momentum().eta()) > 0.82 &&
          std::abs(simMuon->momentum().eta()) < 1.24 && simMuon->momentum().pt() >= 3.) {
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
      //TODO choose a condition, to print the desired candidates
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
  } else {
    //TODO choose a condition, to print the desired candidates
    // an example of a simple cut, only on the canidate pt
    /*
    for (auto& finalCandidate : *finalCandidates) {
      if (finalCandidate.hwPt() < 41) {  //  finalCandidate.hwQual() >= 1  41
        dump = true;
      }
    } */
    //!!!!!!!!!!!!!!!!!!!!!!!! TODO dumps all events!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dump = true;  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  }

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
          << " bx " << bx << " hwPt " << finalCandidate.hwPt() << " hwUPt " << finalCandidate.hwPtUnconstrained()
          << " hwSign " << finalCandidate.hwSign() << " hwQual " << finalCandidate.hwQual() << " hwEta " << std::setw(4)
          << finalCandidate.hwEta() << std::setw(4) << " hwPhi " << finalCandidate.hwPhi() << "    eta " << std::setw(9)
          << (finalCandidate.hwEta() * 0.010875) << " phi " << std::setw(9) << globalPhi << " " << layerHitBits
          << " processor " << OmtfName(finalCandidate.processor(), finalCandidate.trackFinderType(), omtfConfig)
          << std::endl;

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
    OmtfName board(iProc, omtfConfig);

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
                stub->phiHw, OMTFinputMaker::getProcessorPhiZero(omtfConfig, iProc % omtfConfig->nProcessors()));
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
          edm::LogVerbatim("l1tOmtfEventPrint") << algoMuon->getGpResultConstr();
          if (algoMuon->getGpResultUnconstr().isValid())
            edm::LogVerbatim("l1tOmtfEventPrint")
                << "GpResultUnconstr " << algoMuon->getGoldenPaternUnconstr()->key() << "\n"
                << algoMuon->getGpResultUnconstr() << std::endl;

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

      {
        edm::LogVerbatim("l1tOmtfEventPrint") << std::endl << std::endl << "\ngb_test " << board.name() << std::endl;
        for (auto& algoMuon : algoMuonsInProcs[iProc]) {
          edm::LogVerbatim("l1tOmtfEventPrint")
              << "     (" << std::setw(5) << algoMuon->getHwPatternNumConstr() << "," << std::setw(5)
              << algoMuon->getHwPatternNumUnconstr() << ","

              << std::setw(5) << algoMuon->getGpResultConstr().getFiredLayerCnt() << "," << std::setw(5)
              << algoMuon->getGpResultUnconstr().getFiredLayerCnt()
              << ","

              //in the FW there is LSB added for some reason, therefore we multiply by 2 here
              << std::setw(6) << algoMuon->getGpResultConstr().getFiredLayerBits() * 2 << "," << std::setw(6)
              << algoMuon->getGpResultUnconstr().getFiredLayerBits() * 2 << ","

              << std::setw(5) << algoMuon->getGpResultConstr().getPdfSum() << "," << std::setw(5)
              << algoMuon->getGpResultUnconstr().getPdfSumUnconstr() << ","

              << std::setw(5) << algoMuon->getGpResultConstr().getPhi() << "," << std::setw(5)
              << algoMuon->getGpResultUnconstr().getPhi()
              << ","

              //<<std::setw(5)<<OMTFConfiguration::eta2Bits(std::abs(algoMuon->getEtaHw()))<<", "
              << std::setw(5) << algoMuon->getEtaHw()
              << ", "

              //<<std::setw(5)<<omtfConfig->getRefToLogicNumber()[algoMuon->getRefLayer()]<<""
              << std::setw(5) << algoMuon->getRefLayer() << ""

              << "), " << std::endl;
        }
        edm::LogVerbatim("l1tOmtfEventPrint") << "\ngbCandidates" << std::endl;
        for (auto& gbCandidate : gbCandidatesInProcs[iProc])
          edm::LogVerbatim("l1tOmtfEventPrint")
              << "     (" << std::setw(5) << gbCandidate->getPtConstr() << "," << std::setw(5)
              << gbCandidate->getPtUnconstr() << ","

              << std::setw(5) << gbCandidate->getGpResultConstr().getFiredLayerCnt() << "," << std::setw(5)
              << gbCandidate->getGpResultUnconstr().getFiredLayerCnt()
              << ","

              //in the FW there is LSB added for some reason, therefore we multiply by 2 here
              << std::setw(6) << gbCandidate->getGpResultConstr().getFiredLayerBits() * 2 << "," << std::setw(6)
              << gbCandidate->getGpResultUnconstr().getFiredLayerBits() * 2 << ","

              << std::setw(5) << gbCandidate->getGpResultConstr().getPdfSum() << "," << std::setw(5)
              << gbCandidate->getGpResultUnconstr().getPdfSumUnconstr() << ","

              << std::setw(5) << gbCandidate->getGpResultConstr().getPhi() << "," << std::setw(5)
              << gbCandidate->getGpResultUnconstr().getPhi()
              << ","

              //<<std::setw(5)<<OMTFConfiguration::eta2Bits(std::abs(gbCandidate->getEtaHw()))<<", "
              << std::setw(5) << gbCandidate->getEtaHw()
              << ", "

              //<<std::setw(5)<<omtfConfig->getRefToLogicNumber()[gbCandidate->getRefLayer()]<<""
              << std::setw(5) << gbCandidate->getRefLayer() << ""

              << "), "
              << " -- getFiredLayerBits " << std::setw(5) << gbCandidate->getGpResultConstr().getFiredLayerBits()
              << std::endl;

        edm::LogVerbatim("l1tOmtfEventPrint") << "finalCandidates " << std::endl;

        std::ostringstream ostr;
        if (finalCandidates->size(0) > 0) {
          int iMu = 1;
          for (auto finalCandidateIt = finalCandidates->begin(0); finalCandidateIt != finalCandidates->end(0);
               finalCandidateIt++) {
            auto& finalCandidate = *finalCandidateIt;

            auto omtfName = OmtfName(finalCandidate.processor(), finalCandidate.trackFinderType(), omtfConfig);

            if (omtfName == board.name()) {
              int layerHits = (int)finalCandidate.trackAddress().at(0);
              std::bitset<18> layerHitBits(layerHits);

              unsigned int trackAddr = finalCandidate.trackAddress().at(0);
              unsigned int uPt = finalCandidate.hwPtUnconstrained();
              //if(uPt == 0) uPt = 5; //TODO remove when fixed in the FW
              trackAddr = (uPt << 18) + trackAddr;

              ostr << "M" << iMu << ":" << std::setw(4) << finalCandidate.hwPt() << "," << std::setw(4)
                   << finalCandidate.hwQual() << "," << std::setw(4) << finalCandidate.hwPhi() << "," << std::setw(4)
                   << std::abs(finalCandidate.hwEta())
                   << ","
                   //<<std::setw(10)<< finalCandidate.trackAddress().at(0)<<""
                   << std::setw(10) << trackAddr << "," << std::setw(4) << 0 << ","                 //Halo
                   << std::setw(4) << finalCandidate.hwSign() << "," << std::setw(4) << 1 << "; ";  //ChValid
              //<< " -- uPt " << std::setw(10) << uPt << " firedLayers " << finalCandidate.trackAddress().at(0);

              //<<std::setw(5)<< finalCandidate.hwPtUnconstrained()<<","

              //<<std::setw(9)<< layerHitBits<<","
              //<<std::setw(6)<< layerHits<<","
              iMu++;
            }
          }
          for (; iMu <= 3; iMu++)
            ostr << "M" << iMu << ":   0,   0,   0,   0,         0,   0,   0,   0; ";
          edm::LogVerbatim("l1tOmtfEventPrint") << ostr.str() << std::endl;
        }
      }

      edm::LogVerbatim("l1tOmtfEventPrint") << std::endl;
    }
  }

  edm::LogVerbatim("l1tOmtfEventPrint") << std::endl;
}

void EventCapture::endJob() {
  if (stubsSimHitsMatcher)
    stubsSimHitsMatcher->endJob();
}
