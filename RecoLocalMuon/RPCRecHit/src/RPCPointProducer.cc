// -*- C++ -*-
//
// Package:    RPCPointProducer
// Class:      RPCPointProducer
// 
/**\class RPCPointProducer RPCPointProducer.cc Analysis/RPCPointProducer/src/RPCPointProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Camilo Andres Carrillo Montoya
//         Created:  Wed Sep 16 14:56:18 CEST 2009
//
//

#include "RecoLocalMuon/RPCRecHit/interface/RPCPointProducer.h"

// system include files

#include <memory>
#include <ctime>

// user include files

RPCPointProducer::RPCPointProducer(const edm::ParameterSet& iConfig) :
    cscSegments(    consumes<CSCSegmentCollection>(iConfig.getParameter<edm::InputTag>("cscSegments"))),
    dt4DSegments(   consumes<DTRecSegment4DCollection>(iConfig.getParameter<edm::InputTag>("dt4DSegments"))),
    tracks(         consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
    tracks_(        iConfig.getParameter<edm::InputTag>("tracks")),
    debug(          iConfig.getUntrackedParameter<bool>("debug",false)),
    incldt(         iConfig.getUntrackedParameter<bool>("incldt",true)),
    inclcsc(        iConfig.getUntrackedParameter<bool>("inclcsc",true)),
    incltrack(      iConfig.getUntrackedParameter<bool>("incltrack",true)),
    MinCosAng(      iConfig.getUntrackedParameter<double>("MinCosAng",0.95)),
    MaxD(           iConfig.getUntrackedParameter<double>("MaxD",0.95)),
    MaxDrb4(        iConfig.getUntrackedParameter<double>("MaxDrb4",150.)),
    ExtrapolatedRegion(iConfig.getUntrackedParameter<double>("ExtrapolatedRegion",0.5)),
    trackTransformerParam(iConfig.getParameter<edm::ParameterSet>("TrackTransformer"))
{

  produces<RPCRecHitCollection>("RPCDTExtrapolatedPoints");
  produces<RPCRecHitCollection>("RPCCSCExtrapolatedPoints");
  produces<RPCRecHitCollection>("RPCTrackExtrapolatedPoints");
    
}


RPCPointProducer::~RPCPointProducer(){

}

void RPCPointProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  /*
  struct timespec start_time, stop_time;
  time_t fs;
  time_t fn;
  time_t ls;
  time_t ln;
  clock_gettime(CLOCK_REALTIME, &start_time);  
  */

  if(incldt){
    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    iEvent.getByToken(dt4DSegments, all4DSegments);
      
    if (MuonGeometryWatcher.check(iSetup)) TheDTObjectsMap_->fillObjectMapDT(iSetup);
      
    if(all4DSegments.isValid()){
      DTSegtoRPC DTClass(all4DSegments,iSetup,iEvent, debug, ExtrapolatedRegion, TheDTObjectsMap_);
      std::auto_ptr<RPCRecHitCollection> TheDTPoints(DTClass.thePoints());     
      iEvent.put(TheDTPoints,"RPCDTExtrapolatedPoints"); 
    }else{
      LogDebug("RPCPointProducer") <<  std::cout<<"RPCHLT Invalid DTSegments collection"<<std::endl;
    }
  }

  if(inclcsc){
    edm::Handle<CSCSegmentCollection> allCSCSegments;
    iEvent.getByToken(cscSegments, allCSCSegments);
    
      if (MuonGeometryWatcher.check(iSetup)) {
          std::cout << "test  geom" << std::endl;
          TheCSCObjectsMap_->fillObjectMapCSC(iSetup);
      }
      
    if(allCSCSegments.isValid()){
      CSCSegtoRPC CSCClass(allCSCSegments,iSetup,iEvent, debug, ExtrapolatedRegion, TheCSCObjectsMap_);
      std::auto_ptr<RPCRecHitCollection> TheCSCPoints(CSCClass.thePoints());  
      iEvent.put(TheCSCPoints,"RPCCSCExtrapolatedPoints"); 
    }else{
      LogDebug("RPCPointProducer") << std::cout<<"RPCHLT Invalid CSCSegments collection"<<std::endl;
    }
  }
  if(incltrack){
    edm::Handle<reco::TrackCollection> alltracks;
    iEvent.getByToken(tracks,alltracks);
      
    if (MuonGeometryWatcher.check(iSetup)) {
        TheDTtrackObjectsMap_->fillObjectMapDT(iSetup);
        TheCSCtrackObjectsMap_->fillObjectMapCSC(iSetup);
    }
      
    if(!(alltracks->empty())){
      TracktoRPC TrackClass(alltracks,iSetup,iEvent,trackTransformerParam,tracks_, TheDTtrackObjectsMap_, TheCSCtrackObjectsMap_);
      std::auto_ptr<RPCRecHitCollection> TheTrackPoints(TrackClass.thePoints());
      iEvent.put(TheTrackPoints,"RPCTrackExtrapolatedPoints");
    }else{
      LogDebug("RPCPointProducer") <<"RPCHLT Invalid Tracks collection"<<std::endl;
    }
  }
 
}

void  RPCPointProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup){
    TheDTObjectsMap_  = new ObjectMap(iSetup);
    TheCSCObjectsMap_ = new ObjectMapCSC(iSetup);
    TheDTtrackObjectsMap_ = new ObjectMap2(iSetup);
    TheCSCtrackObjectsMap_ = new ObjectMap2CSC(iSetup);

    
}


void  RPCPointProducer::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup){

    delete TheDTObjectsMap_;
    delete TheCSCObjectsMap_;
    delete TheDTtrackObjectsMap_;
    delete TheCSCtrackObjectsMap_;
    
    
}

