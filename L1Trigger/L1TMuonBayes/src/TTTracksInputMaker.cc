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
  }
}

TTTracksInputMaker::~TTTracksInputMaker() {
  // TODO Auto-generated destructor stub
}

TrackingTriggerTracks TTTracksInputMaker::loadTTTracks(const edm::Event &event, const edm::ParameterSet& edmCfg, const ProcConfigurationBase* procConf) {
  TrackingTriggerTracks ttTracks;
  //cout<<__FUNCTION__<<":"<<__LINE__<<endl;

  if(ttTracksSource == SIM_TRACKS) {
    edm::Handle<edm::SimTrackContainer> simTraksHandle;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("g4SimTrackSrc"), simTraksHandle);
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" simTks.size() "<<simTks->size()<<std::endl;
    for (unsigned int iSimTrack = 0; iSimTrack != simTraksHandle->size(); iSimTrack++ ) {
      edm::Ptr< SimTrack > simTrackPtr(simTraksHandle, iSimTrack);
      if ( (abs(simTrackPtr->type()) == 13  ||  abs(simTrackPtr->type()) == 1000015) && simTrackPtr->momentum().pt() > 2.5) { //TODO 1000015 is stau
        auto ttTrack = std::make_shared<TrackingTriggerTrack>(simTrackPtr);
        ttTrack->setSimBeta(simTrackPtr->momentum().Beta());

        addTTTrack(ttTracks, ttTrack, procConf);
        //if(ttTrack->getPt() > 20)

        LogTrace("omtfEventPrintout")<<__FUNCTION__<<":"<<__LINE__<<" sim.type() "<<simTrackPtr->type()<<" genpartIndex "<<simTrackPtr->genpartIndex()
                <<" Beta() "<<simTrackPtr->momentum().Beta()<<" added track "<<*ttTrack<<std::endl;
      }
    }

  }
  else if(ttTracksSource == L1_TRACKER) {
    edm::Handle< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > tTTrackHandle;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("L1TrackInputTag"), tTTrackHandle);
    //cout << __FUNCTION__<<":"<<__LINE__ << " LTTTrackHandle->size() "<<tTTrackHandle->size() << endl;

    for (unsigned int iTTTrack = 0; iTTTrack != tTTrackHandle->size(); iTTTrack++ ) {
      edm::Ptr< TTTrack< Ref_Phase2TrackerDigi_ > > ttTrackPtr(tTTrackHandle, iTTTrack);
      auto ttTrack = std::make_shared<TrackingTriggerTrack>(ttTrackPtr, l1Tk_nPar);

      if(ttTrackPtr->getStubRefs().size() >= l1Tk_minNStub) //TODO is this cut possible to apply in the firmware? there should be "Hit mask" so should be used whenever available
        addTTTrack(ttTracks, ttTrack, procConf);

      //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<*iterL1Track<<" Momentum "<<iterL1Track->getMomentum(l1Tk_nPar)<<" RInv "<<iterL1Track->getRInv(l1Tk_nPar)<<endl;
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
