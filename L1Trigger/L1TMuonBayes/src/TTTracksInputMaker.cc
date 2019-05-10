/*
 * TTTracksInputMaker.cc
 *
 *  Created on: Jan 31, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonBayes/interface/TTTracksInputMaker.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

TTTracksInputMaker::TTTracksInputMaker(const edm::ParameterSet& edmCfg) {
  if(edmCfg.exists("ttTracksSource") ) {
    std::string trackSrc = edmCfg.getParameter<std::string>("ttTracksSource");
    if(trackSrc == "SIM_TRACKS")
      ttTracksSource = SIM_TRACKS;
    else if(trackSrc == "L1_TRACKER") {
      ttTracksSource = L1_TRACKER;
      if(edmCfg.exists("l1Tk_nPar") ) {
        l1Tk_nPar = edmCfg.getParameter<int>("l1Tk_nPar");
      }
      if(edmCfg.exists("l1Tk_minNStub") ) {
        l1Tk_minNStub = edmCfg.getParameter<int>("l1Tk_minNStub");
      }
    }
    if(trackSrc == "TRACKING_PARTICLES")
      ttTracksSource = TRACKING_PARTICLES;
  }
}

TTTracksInputMaker::~TTTracksInputMaker() {

}

TrackingTriggerTracks TTTracksInputMaker::loadTTTracks(const edm::Event &event, int bx, const edm::ParameterSet& edmCfg, const ProcConfigurationBase* procConf) {
  TrackingTriggerTracks ttTracks;
  //cout<<__FUNCTION__<<":"<<__LINE__<<endl;

  if(ttTracksSource == SIM_TRACKS) {
    edm::Handle<edm::SimTrackContainer> simTracksHandle;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("g4SimTrackSrc"), simTracksHandle);
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" simTks.size() "<<simTks->size()<<std::endl;
    for (unsigned int iSimTrack = 0; iSimTrack != simTracksHandle->size(); iSimTrack++ ) {
      edm::Ptr< SimTrack > simTrackPtr(simTracksHandle, iSimTrack);

      if(simTrackPtr->eventId().bunchCrossing() == bx) {
        if ( (abs(simTrackPtr->type()) == 13  ||  abs(simTrackPtr->type()) == 1000015) && simTrackPtr->momentum().pt() > 2.5) { //TODO 1000015 is stau
          auto ttTrack = std::make_shared<TrackingTriggerTrack>(simTrackPtr);
          ttTrack->setSimBeta(simTrackPtr->momentum().Beta());

          addTTTrack(ttTracks, ttTrack, procConf);
          //if(ttTrack->getPt() > 20)

          LogTrace("omtfEventPrintout")<<__FUNCTION__<<":"<<__LINE__<<" bx "<<bx<<" adding ttTrack from simTrack: sim.type() "<<simTrackPtr->type()<<" genpartIndex "<<simTrackPtr->genpartIndex()
                    <<" Beta() "<<simTrackPtr->momentum().Beta()<<" added track "<<*ttTrack<<std::endl;
        }
      }
    }
  }
  else if(bx != 0 || ttTracksSource == TRACKING_PARTICLES) {
    edm::Handle< std::vector< TrackingParticle > > trackingParticlesHandle;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("TrackingParticleInputTag"), trackingParticlesHandle);
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" simTks.size() "<<simTks->size()<<std::endl;
    for (unsigned int iTP = 0; iTP != trackingParticlesHandle->size(); iTP++ ) {
      edm::Ptr< TrackingParticle > trackingParticlePtr(trackingParticlesHandle, iTP);

      if(trackingParticlePtr->eventId().bunchCrossing() == bx) {//to emulate the trigger rules we should process every track not only the muons!!!!
        //if ( (abs(trackingParticlePtr->pdgId()) == 13  ||  abs(trackingParticlePtr->pdgId()) == 1000015) && trackingParticlePtr->pt() > 2.5) //TODO 1000015 is stau
        if(trackingParticlePtr->pt() > 2.4 && abs(trackingParticlePtr->eta() ) < 2.4) //todo move values to config
        {
          auto ttTrack = std::make_shared<TrackingTriggerTrack>(trackingParticlePtr);
          ttTrack->setSimBeta(trackingParticlePtr->p4().Beta());

          addTTTrack(ttTracks, ttTrack, procConf);
          //if(ttTrack->getPt() > 20)

          //LogTrace("omtfEventPrintout")<<__FUNCTION__<<":"<<__LINE__<<" bx "<<bx<<" adding ttTrack from TrackingParticle: pdgId "<<trackingParticlePtr->pdgId()<<" genParticles().size() "<<trackingParticlePtr->genParticles().size()
          //          <<" Beta() "<<trackingParticlePtr->p4().Beta()<<" added track "<<*ttTrack<<std::endl;
        }
      }
    }
  }
  else if(ttTracksSource == L1_TRACKER) {
    edm::Handle< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > tTTrackHandle;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("L1TrackInputTag"), tTTrackHandle);
    //cout << __FUNCTION__<<":"<<__LINE__ << " LTTTrackHandle->size() "<<tTTrackHandle->size() << endl;

    for (unsigned int iTTTrack = 0; iTTTrack != tTTrackHandle->size(); iTTTrack++ ) {
      edm::Ptr< TTTrack< Ref_Phase2TrackerDigi_ > > ttTrackPtr(tTTrackHandle, iTTTrack);
      //TODO is there bx in the ttTrackPtr? Or the emulator works only in the BX 0, thus no tracks in other BXes?
      auto ttTrack = std::make_shared<TrackingTriggerTrack>(ttTrackPtr, l1Tk_nPar);

      if(ttTrackPtr->getStubRefs().size() >= l1Tk_minNStub) //TODO is this cut possible to apply in the firmware? there should be "Hit mask" so should be used whenever available
        addTTTrack(ttTracks, ttTrack, procConf);

      //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<*iterL1Track<<" Momentum "<<iterL1Track->getMomentum(l1Tk_nPar)<<" RInv "<<iterL1Track->getRInv(l1Tk_nPar)<<endl;
      //LogTrace("omtfEventPrintout")<<__FUNCTION__<<":"<<__LINE__<<" bx "<<bx<<" adding ttTrack from TTTrack: "<<" added track "<<*ttTrack<<std::endl;
    }
  }
  //cout<<__FUNCTION__<<":"<<__LINE__<<" ttTracks.size() "<<ttTracks.size()<<endl;

  return ttTracks;
}

void TTTracksInputMaker::addTTTrack(TrackingTriggerTracks& ttTracks, std::shared_ptr<TrackingTriggerTrack>& ttTrack, const ProcConfigurationBase* procConf) {
  ttTrack->setPhiHw(procConf->getProcScalePhi(ttTrack->getPhi() ));
  ttTrack->setEtaHw(procConf->etaToHwEta( ttTrack->getEta() ));
  ttTrack->setPtHw(procConf->ptGevToHw(ttTrack->getPt() ) );

  ttTrack->setPtBin(procConf->ptHwToPtBin(ttTrack->getPtHw() ) ); //TODO when real hardware skale available, move to this function, and implement properly ptHwToPtBin
  //ttTrack->setPtBin(procConf->ptGeVToPtBin(ttTrack->getPt() ) );
  ttTrack->setEtaBin(procConf->etaHwToEtaBin(ttTrack->getEtaHw() ) );

  //std::cout << __FUNCTION__<<":"<<__LINE__ <<" pt "<<ttTrack->getPt()<<" PtHw "<<ttTrack->getPtHw()<<" PtBin "<<ttTrack->getPtBin()<<std::endl;
  ttTracks.emplace_back(ttTrack);
}
