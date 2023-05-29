/*
 * StubsSimHitsMatching.cc
 *
 *  Created on: Jan 13, 2021
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/StubsSimHitsMatcher.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OmtfName.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/AngleConverterBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLink.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <cmath>

StubsSimHitsMatcher::StubsSimHitsMatcher(const edm::ParameterSet& edmCfg,
                                         const OMTFConfiguration* omtfConfig,
                                         const MuonGeometryTokens& muonGeometryTokens)
    : omtfConfig(omtfConfig), muonGeometryTokens(muonGeometryTokens) {
  rpcSimHitsInputTag = edmCfg.getParameter<edm::InputTag>("rpcSimHitsInputTag");
  cscSimHitsInputTag = edmCfg.getParameter<edm::InputTag>("cscSimHitsInputTag");
  dtSimHitsInputTag = edmCfg.getParameter<edm::InputTag>("dtSimHitsInputTag");

  rpcDigiSimLinkInputTag = edmCfg.getParameter<edm::InputTag>("rpcDigiSimLinkInputTag");
  cscStripDigiSimLinksInputTag = edmCfg.getParameter<edm::InputTag>("cscStripDigiSimLinksInputTag");
  dtDigiSimLinksInputTag = edmCfg.getParameter<edm::InputTag>("dtDigiSimLinksInputTag");

  trackingParticleTag = edmCfg.getParameter<edm::InputTag>("trackingParticleTag");

  edm::Service<TFileService> fs;
  TFileDirectory subDir = fs->mkdir("StubsSimHitsMatcher");

  allMatchedTracksPdgIds = subDir.make<TH1I>("allMatchedTracksPdgIds", "allMatchedTracksPdgIds", 3, 0, 3);
  allMatchedTracksPdgIds->SetCanExtend(TH1::kAllAxes);
  bestMatchedTracksPdgIds = subDir.make<TH1I>("bestMatchedTracksPdgIds", "bestMatchedTracksPdgIds", 3, 0, 3);
  bestMatchedTracksPdgIds->SetCanExtend(TH1::kAllAxes);

  stubsInLayersCntByPdgId = subDir.make<TH2I>("stubsInLayersCntByPdgId",
                                              "stubsInLayersCntByPdgId;;OMTF layer;",
                                              3,
                                              0,
                                              3,
                                              omtfConfig->nLayers(),
                                              -.5,
                                              omtfConfig->nLayers() - 0.5);
  stubsInLayersCntByPdgId->SetCanExtend(TH1::kAllAxes);

  firedLayersCntByPdgId = subDir.make<TH2I>("firedLayersCntByPdgId",
                                            "firedLayersCntByPdgId",
                                            3,
                                            0,
                                            3,
                                            omtfConfig->nLayers(),
                                            -.5,
                                            omtfConfig->nLayers() - 0.5);
  firedLayersCntByPdgId->SetCanExtend(TH1::kAllAxes);

  ptByPdgId = subDir.make<TH2I>("ptByPdgId", "ptByPdgId bestMatched;;pT [GeV];#", 3, 0, 3, 20, 0, 10);
  ptByPdgId->SetCanExtend(TH1::kAllAxes);

  rhoByPdgId = subDir.make<TH2I>("rhoByPdgId", "rhoByPdgId bestMatched;;Rho [cm];#", 3, 0, 3, 50, 0, 500);
  rhoByPdgId->SetCanExtend(TH1::kAllAxes);
}

StubsSimHitsMatcher::~StubsSimHitsMatcher() {}

void StubsSimHitsMatcher::beginRun(edm::EventSetup const& eventSetup) {
  if (muonGeometryRecordWatcher.check(eventSetup)) {
    _georpc = eventSetup.getHandle(muonGeometryTokens.rpcGeometryEsToken);
    _geocsc = eventSetup.getHandle(muonGeometryTokens.cscGeometryEsToken);
    _geodt = eventSetup.getHandle(muonGeometryTokens.dtGeometryEsToken);
  }
}

void StubsSimHitsMatcher::match(const edm::Event& iEvent,
                                const l1t::RegionalMuonCand* omtfCand,
                                const AlgoMuonPtr& procMuon,
                                std::ostringstream& ostr) {
  ostr << "stubsSimHitsMatching ---------------" << std::endl;

  edm::Handle<edm::PSimHitContainer> rpcSimHitsHandle;
  iEvent.getByLabel(rpcSimHitsInputTag, rpcSimHitsHandle);
  ostr << "rpcSimHitsHandle: size: " << rpcSimHitsHandle->size() << std::endl;

  edm::Handle<edm::PSimHitContainer> dtSimHitsHandle;
  iEvent.getByLabel(dtSimHitsInputTag, dtSimHitsHandle);
  ostr << std::endl << "dtSimHitsHandle: size: " << dtSimHitsHandle->size() << std::endl;

  edm::Handle<edm::PSimHitContainer> cscSimHitsHandle;
  iEvent.getByLabel(cscSimHitsInputTag, cscSimHitsHandle);
  ostr << std::endl << "cscSimHitsHandle: size: " << cscSimHitsHandle->size() << std::endl;

  edm::Handle<edm::DetSetVector<RPCDigiSimLink> > rpcDigiSimLinkHandle;
  iEvent.getByLabel(rpcDigiSimLinkInputTag, rpcDigiSimLinkHandle);
  ostr << "rpcDigiSimLinkHandle: size: " << rpcDigiSimLinkHandle->size() << std::endl;

  edm::Handle<edm::DetSetVector<StripDigiSimLink> > cscStripDigiSimLinkHandle;
  iEvent.getByLabel(cscStripDigiSimLinksInputTag, cscStripDigiSimLinkHandle);
  ostr << "cscStripDigiSimLinkHandle: size: " << cscStripDigiSimLinkHandle->size() << std::endl;

  edm::Handle<MuonDigiCollection<DTLayerId, DTDigiSimLink> > dtDigiSimLinkHandle;
  iEvent.getByLabel(dtDigiSimLinksInputTag, dtDigiSimLinkHandle);
  //ostr<<"dtDigiSimLinkHandle: size: " << dtDigiSimLinkHandle->size()<<std::endl;

  edm::Handle<TrackingParticleCollection> trackingParticleHandle;
  iEvent.getByLabel(trackingParticleTag, trackingParticleHandle);

  if (procMuon->isValid() && omtfCand) {
    OmtfName board(omtfCand->processor(), omtfCand->trackFinderType());
    auto processorPhiZero = OMTFinputMaker::getProcessorPhiZero(omtfConfig, omtfCand->processor());

    std::set<MatchedTrackInfo> matchedTrackInfos;
    ostr << board.name() << " " << *procMuon << std::endl;

    auto& gpResult = procMuon->getGpResult();
    for (unsigned int iLogicLayer = 0; iLogicLayer < gpResult.getStubResults().size(); ++iLogicLayer) {
      auto& stub = gpResult.getStubResults()[iLogicLayer].getMuonStub();
      if (stub && gpResult.isLayerFired(iLogicLayer)) {
        if (omtfConfig->isBendingLayer(iLogicLayer))
          continue;

        DetId stubDetId(stub->detId);
        if (stubDetId.det() != DetId::Muon) {
          edm::LogError("l1tOmtfEventPrint")
              << "!!!!!!!!!!!!!!!!!!!!!!!!  PROBLEM: hit in unknown Det, detID: " << stubDetId.det() << std::endl;
          continue;
        }

        auto stubGlobalPhi = omtfConfig->procHwPhiToGlobalPhi(stub->phiHw, processorPhiZero);
        ostr << (*stub) << "\nstubGlobalPhi " << stubGlobalPhi << std::endl;

        switch (stubDetId.subdetId()) {
          case MuonSubdetId::RPC: {
            RPCDetId rpcDetId(stubDetId);

            for (auto& simHit : *(rpcSimHitsHandle.product())) {
              if (stubDetId.rawId() == simHit.detUnitId()) {
                const RPCRoll* roll = _georpc->roll(rpcDetId);
                auto strip = roll->strip(simHit.localPosition());
                double simHitStripGlobalPhi = (roll->toGlobal(roll->centreOfStrip((int)strip))).phi();

                if (abs(stubGlobalPhi - simHitStripGlobalPhi) < 0.02) {
                  matchedTrackInfos.insert(MatchedTrackInfo(simHit.eventId().event(), simHit.trackId()));
                }

                ostr << " simHitStripGlobalPhi " << std::setw(10) << simHitStripGlobalPhi << " strip " << strip
                     << " particleType: " << simHit.particleType() << " event: " << simHit.eventId().event()
                     << " trackId " << simHit.trackId() << " processType " << simHit.processType() << " detUnitId "
                     << simHit.detUnitId() << " "
                     << rpcDetId
                     //<<" phiAtEntry "<<simHit.phiAtEntry()
                     //<<" thetaAtEntry "<<simHit.thetaAtEntry()
                     //<<" localPosition: phi "<<simHit.localPosition().phi()<<" eta "<<simHit.localPosition().eta()
                     << " entryPoint: x " << std::setw(10) << simHit.entryPoint().x() << " y " << std::setw(10)
                     << simHit.entryPoint().y() << " timeOfFlight " << simHit.timeOfFlight() << std::endl;
              }
            }

            auto rpcDigiSimLinkDetSet = rpcDigiSimLinkHandle.product()->find(stub->detId);

            if (rpcDigiSimLinkDetSet != rpcDigiSimLinkHandle.product()->end()) {
              ostr << "rpcDigiSimLinkDetSet: detId " << rpcDigiSimLinkDetSet->detId() << " size "
                   << rpcDigiSimLinkDetSet->size() << "\n";
              for (auto& rpcDigiSimLink : *rpcDigiSimLinkDetSet) {
                const RPCRoll* roll = _georpc->roll(rpcDetId);
                auto strip = rpcDigiSimLink.getStrip();
                double simHitStripGlobalPhi = (roll->toGlobal(roll->centreOfStrip((int)strip))).phi();

                if (abs(stubGlobalPhi - simHitStripGlobalPhi) < 0.02) {
                  auto matchedTrackInfo = matchedTrackInfos.insert(
                      MatchedTrackInfo(rpcDigiSimLink.getEventId().event(), rpcDigiSimLink.getTrackId()));
                  matchedTrackInfo.first->matchedDigiCnt.at(iLogicLayer)++;
                }

                ostr << " simHitStripGlobalPhi " << std::setw(10) << simHitStripGlobalPhi << " strip " << strip
                     << " particleType: " << rpcDigiSimLink.getParticleType()
                     << " event: " << rpcDigiSimLink.getEventId().event() << " trackId " << rpcDigiSimLink.getTrackId()
                     << " processType " << rpcDigiSimLink.getProcessType() << " detUnitId "
                     << rpcDigiSimLink.getDetUnitId() << " "
                     << rpcDetId
                     //<<" phiAtEntry "<<simHit.phiAtEntry()
                     //<<" thetaAtEntry "<<simHit.thetaAtEntry()
                     //<<" localPosition: phi "<<simHit.localPosition().phi()<<" eta "<<simHit.localPosition().eta()
                     << " entryPoint: x " << std::setw(10) << rpcDigiSimLink.getEntryPoint().x() << " y "
                     << std::setw(10) << rpcDigiSimLink.getEntryPoint().y() << " timeOfFlight "
                     << rpcDigiSimLink.getTimeOfFlight() << std::endl;
              }
            }

            break;
          }  //----------------------------------------------------------------------
          case MuonSubdetId::DT: {
            //DTChamberId dt(stubDetId);
            for (auto& simHit : *(dtSimHitsHandle.product())) {
              const DTLayer* layer = _geodt->layer(DTLayerId(simHit.detUnitId()));
              const DTChamber* chamber = layer->chamber();
              if (stubDetId.rawId() == chamber->id().rawId()) {
                //auto strip = layer->geometry()->strip(simHit.localPosition());
                auto simHitGlobalPoint = layer->toGlobal(simHit.localPosition());

                ostr << " simHitGlobalPoint.phi " << std::setw(10)
                     << simHitGlobalPoint.phi()
                     //<<" strip "<<strip
                     << " particleType: " << simHit.particleType() << " event: " << simHit.eventId().event()
                     << " trackId " << simHit.trackId() << " processType " << simHit.processType() << " detUnitId "
                     << simHit.detUnitId() << " "
                     << layer->id()
                     //<<" phiAtEntry "<<simHit.phiAtEntry()
                     //<<" thetaAtEntry "<<simHit.thetaAtEntry()
                     //<<" localPosition: phi "<<simHit.localPosition().phi()<<" eta "<<simHit.localPosition().eta()
                     << " localPosition: x " << std::setw(10) << simHit.localPosition().x() << " y " << std::setw(10)
                     << simHit.localPosition().y() << " timeOfFlight " << simHit.timeOfFlight() << std::endl;
              }
            }

            auto chamber = _geodt->chamber(DTLayerId(stub->detId));
            for (auto superlayer : chamber->superLayers()) {
              if (superlayer->id().superLayer() == 2)  //we skip the theta layer
                continue;
              for (auto layer : superlayer->layers()) {
                auto dtDigiSimLinks = dtDigiSimLinkHandle.product()->get(layer->id());
                ostr << "dt layer " << layer->id() << "\n";
                auto dtDigiSimLink = dtDigiSimLinks.first;

                for (; dtDigiSimLink != dtDigiSimLinks.second; dtDigiSimLink++) {
                  //const RPCRoll* roll = _georpc->roll(rpcDetId);
                  auto wire = dtDigiSimLink->wire();

                  auto wireX = layer->specificTopology().wirePosition(wire);

                  LocalPoint point(wireX, 0, 0);
                  auto digiWireGlobal = layer->toGlobal(point);

                  if (abs(stubGlobalPhi - digiWireGlobal.phi()) < 0.03) {
                    auto matchedTrackInfo = matchedTrackInfos.insert(
                        MatchedTrackInfo(dtDigiSimLink->eventId().event(), dtDigiSimLink->SimTrackId()));
                    matchedTrackInfo.first->matchedDigiCnt.at(iLogicLayer)++;
                  }

                  ostr
                      << " digiWireGlobalPhi " << std::setw(10) << digiWireGlobal.phi() << " wire "
                      << wire
                      //<<" particleType: "<<cscDigiSimLink.getParticleType()
                      << " event: " << dtDigiSimLink->eventId().event() << " trackId "
                      << dtDigiSimLink->SimTrackId()
                      //<<" CFposition "<<cscDigiSimLink.CFposition() is 0
                      //the rest is not available in the StripDigiSimLink, maybe the SimHit must be found in the  CrossingFrame vector(???) with CFposition()
                      //<<" processType "<<cscDigiSimLink.getProcessType()
                      //<<" detUnitId "<<cscDigiSimLink.getDetUnitId()<<" "<<rpcDetId
                      //<<" phiAtEntry "<<simHit.phiAtEntry()
                      //<<" thetaAtEntry "<<simHit.thetaAtEntry()
                      //<<" localPosition: phi "<<simHit.localPosition().phi()<<" eta "<<simHit.localPosition().eta()
                      //<<" entryPoint: x "<<std::setw(10)<<rpcDigiSimLink.getEntryPoint().x()<<" y "<<std::setw(10)<<rpcDigiSimLink.getEntryPoint().y()
                      //<<" timeOfFlight "<<rpcDigiSimLink.getTimeOfFlight()
                      << std::endl;
                }
              }
            }

            break;
          }  //----------------------------------------------------------------------
          case MuonSubdetId::CSC: {
            //CSCDetId csc(stubDetId);
            for (auto& simHit : *(cscSimHitsHandle.product())) {
              const CSCLayer* layer = _geocsc->layer(CSCDetId(simHit.detUnitId()));
              auto chamber = layer->chamber();
              if (stubDetId.rawId() == chamber->id().rawId()) {
                auto simHitStrip = layer->geometry()->strip(simHit.localPosition());
                auto simHitGlobalPoint = layer->toGlobal(simHit.localPosition());
                auto simHitStripGlobalPhi = layer->centerOfStrip(round(simHitStrip)).phi();

                ostr << " simHit: gloablPoint phi " << simHitGlobalPoint.phi() << " stripGlobalPhi "
                     << simHitStripGlobalPhi.phi() << " strip " << simHitStrip
                     << " particleType: " << simHit.particleType() << " event: " << simHit.eventId().event()
                     << " trackId " << simHit.trackId() << " processType " << simHit.processType() << " detUnitId "
                     << simHit.detUnitId() << " "
                     << layer->id()
                     //<<" phiAtEntry "<<simHit.phiAtEntry()
                     //<<" thetaAtEntry "<<simHit.thetaAtEntry()
                     << " timeOfFlight "
                     << simHit.timeOfFlight()
                     //<<" localPosition: phi "<<simHit.localPosition().phi()<<" eta "<<simHit.localPosition().eta()
                     << " x " << simHit.localPosition().x() << " y " << simHit.localPosition().y()

                     << std::endl;
              }
            }

            auto chamber = _geocsc->chamber(CSCDetId(stub->detId));
            for (auto* layer : chamber->layers()) {
              auto cscDigiSimLinkDetSet = cscStripDigiSimLinkHandle.product()->find(layer->id());
              if (cscDigiSimLinkDetSet != cscStripDigiSimLinkHandle.product()->end()) {
                ostr << "cscDigiSimLinkDetSet: detId " << cscDigiSimLinkDetSet->detId() << " " << layer->id()
                     << " size " << cscDigiSimLinkDetSet->size() << "\n";
                for (auto& cscDigiSimLink : *cscDigiSimLinkDetSet) {
                  //const RPCRoll* roll = _georpc->roll(rpcDetId);
                  auto strip = cscDigiSimLink.channel();
                  auto digiStripGlobalPhi = layer->centerOfStrip(strip).phi();

                  if (abs(stubGlobalPhi - digiStripGlobalPhi) < 0.03) {
                    auto matchedTrackInfo = matchedTrackInfos.insert(
                        MatchedTrackInfo(cscDigiSimLink.eventId().event(), cscDigiSimLink.SimTrackId()));
                    matchedTrackInfo.first->matchedDigiCnt.at(iLogicLayer)++;
                  }
                  ostr
                      << " digiStripGlobalPhi " << std::setw(10) << digiStripGlobalPhi << " strip "
                      << strip
                      //<<" particleType: "<<cscDigiSimLink.getParticleType()
                      << " event: " << cscDigiSimLink.eventId().event() << " trackId "
                      << cscDigiSimLink.SimTrackId()
                      //<<" CFposition "<<cscDigiSimLink.CFposition() is 0
                      //the rest is not available in the StripDigiSimLink, maybe the SimHit must be found in the  CrossingFrame vector(???) with CFposition()
                      //<<" processType "<<cscDigiSimLink.getProcessType()
                      //<<" detUnitId "<<cscDigiSimLink.getDetUnitId()<<" "<<rpcDetId
                      //<<" phiAtEntry "<<simHit.phiAtEntry()
                      //<<" thetaAtEntry "<<simHit.thetaAtEntry()
                      //<<" localPosition: phi "<<simHit.localPosition().phi()<<" eta "<<simHit.localPosition().eta()
                      //<<" entryPoint: x "<<std::setw(10)<<rpcDigiSimLink.getEntryPoint().x()<<" y "<<std::setw(10)<<rpcDigiSimLink.getEntryPoint().y()
                      //<<" timeOfFlight "<<rpcDigiSimLink.getTimeOfFlight()
                      << std::endl;
                }
              } else {
                ostr << "cscDigiSimLinkDetSet not found for detId " << layer->id();
              }
            }

            break;
          }  //end of CSC case
        }    //end of switch
        ostr << "" << std::endl;
      }
    }

    ostr << board.name() << " " << *procMuon << std::endl;
    ostr << procMuon->getGpResult() << std::endl << std::endl;

    int maxMatchedStubs = 0;
    const TrackingParticle* bestMatchedPart = nullptr;
    for (auto matchedTrackInfo : matchedTrackInfos) {
      ostr << "matchedTrackInfo eventNum " << matchedTrackInfo.eventNum << " trackId " << matchedTrackInfo.trackId
           << "\n";

      const TrackingParticle* matchedPart = nullptr;
      for (auto& trackingParticle :
           *trackingParticleHandle.product()) {  //finding the trackingParticle corresponding to the matchedTrackInfo
        if (trackingParticle.eventId().event() == matchedTrackInfo.eventNum &&
            trackingParticle.g4Tracks().at(0).trackId() == matchedTrackInfo.trackId) {
          allMatchedTracksPdgIds->Fill(to_string(trackingParticle.pdgId()).c_str(), 1);
          matchedPart = &trackingParticle;

          ostr << "trackingParticle: pdgId " << std::setw(3) << trackingParticle.pdgId() << " event " << std::setw(4)
               << trackingParticle.eventId().event() << " trackId " << std::setw(8)
               << trackingParticle.g4Tracks().at(0).trackId() << " pt " << std::setw(9) << trackingParticle.pt()
               << " eta " << std::setw(9) << trackingParticle.momentum().eta() << " phi " << std::setw(9)
               << trackingParticle.momentum().phi() << std::endl;

          if (trackingParticle.parentVertex().isNonnull()) {
            ostr << "parentVertex Rho " << trackingParticle.parentVertex()->position().Rho() << " z "
                 << trackingParticle.parentVertex()->position().z() << " R "
                 << trackingParticle.parentVertex()->position().R() << " eta "
                 << trackingParticle.parentVertex()->position().eta() << " phi "
                 << trackingParticle.parentVertex()->position().phi() << std::endl;

            for (auto& parentTrack : trackingParticle.parentVertex()->sourceTracks()) {
              ostr << "parentTrack:      pdgId " << std::setw(3) << parentTrack->pdgId() << " event " << std::setw(4)
                   << parentTrack->eventId().event() << " trackId " << std::setw(8)
                   << parentTrack->g4Tracks().at(0).trackId() << " pt " << std::setw(9) << parentTrack->pt() << " eta "
                   << std::setw(9) << parentTrack->momentum().eta() << " phi " << std::setw(9)
                   << parentTrack->momentum().phi() << std::endl;
            }
          }

          break;
        }
      }

      int matchedStubsCnt = 0;
      for (unsigned int iLayer = 0; iLayer < matchedTrackInfo.matchedDigiCnt.size(); iLayer++) {
        ostr << " matchedDigiCnt in layer " << iLayer << " " << matchedTrackInfo.matchedDigiCnt[iLayer] << "\n";
        if (matchedTrackInfo.matchedDigiCnt[iLayer] > 0) {
          matchedStubsCnt++;
          if (matchedPart) {
            string str = to_string(matchedPart->pdgId());
            stubsInLayersCntByPdgId->Fill(str.c_str(), iLayer, 1);
          }
        }
      }
      ostr << std::endl;

      if (maxMatchedStubs < matchedStubsCnt) {
        maxMatchedStubs = matchedStubsCnt;
        bestMatchedPart = matchedPart;
      }
    }
    if (bestMatchedPart) {
      string str = to_string(bestMatchedPart->pdgId());

      bestMatchedTracksPdgIds->Fill(str.c_str(), 1);
      firedLayersCntByPdgId->Fill(str.c_str(), gpResult.getFiredLayerCnt(), 1);
      ptByPdgId->Fill(str.c_str(), bestMatchedPart->pt(), 1);

      if (bestMatchedPart->parentVertex().isNonnull()) {
        rhoByPdgId->Fill(str.c_str(), bestMatchedPart->parentVertex()->position().Rho(), 1);
      }

      ostr << "bestMatchedPart: pdgId " << std::setw(3) << bestMatchedPart->pdgId() << " event " << std::setw(4)
           << bestMatchedPart->eventId().event() << " trackId " << std::setw(8)
           << bestMatchedPart->g4Tracks().at(0).trackId() << "\n\n";
    }
  }
}

void StubsSimHitsMatcher::endJob() {
  /*edm::LogVerbatim("l1tOmtfEventPrint") <<"allMatchedTracksPdgIds "<<std::endl;
  for(auto pdgId : allMatchedTracksPdgIds) {
    edm::LogVerbatim("l1tOmtfEventPrint") <<"pdgId "<<std::setw(8)<<pdgId.first<<" muon candidates "<<pdgId.second<<std::endl;
  }

  edm::LogVerbatim("l1tOmtfEventPrint") <<"bestMatchedTracksPdgIds "<<std::endl;
  for(auto pdgId : bestMatchedTracksPdgIds) {
    edm::LogVerbatim("l1tOmtfEventPrint") <<"pdgId "<<std::setw(8) <<pdgId.first<<" muon candidates "<<pdgId.second<<std::endl;
  }*/
}
