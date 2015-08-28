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
  cscSegments(  consumes<CSCSegmentCollection>(iConfig.getParameter<edm::InputTag>("cscSegments")) ),
  dt4DSegments( consumes<DTRecSegment4DCollection>(iConfig.getParameter<edm::InputTag>("dt4DSegments")) ),
  tracks(       consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks")) ),
  tracks_(      iConfig.getParameter<edm::InputTag>("tracks") ),
  incldt(       iConfig.getUntrackedParameter<bool>("incldt", true) ),
  inclcsc(      iConfig.getUntrackedParameter<bool>("inclcsc", true) ),
  incltrack(    iConfig.getUntrackedParameter<bool>("incltrack", true) ), 
  debug(        iConfig.getUntrackedParameter<bool>("debug", false) ),
  MinCosAng(    iConfig.getUntrackedParameter<double>("MinCosAng", 0.95) ),
  MaxD(         iConfig.getUntrackedParameter<double>("MaxD", 80.) ),
  MaxDrb4(      iConfig.getUntrackedParameter<double>("MaxDrb4", 150.) ),
  ExtrapolatedRegion( iConfig.getUntrackedParameter<double>("ExtrapolatedRegion", 0.5) ),
  trackTransformerParam( iConfig.getParameter<edm::ParameterSet>("TrackTransformer") )
{
  produces<RPCRecHitCollection>("RPCDTExtrapolatedPoints");
  produces<RPCRecHitCollection>("RPCCSCExtrapolatedPoints");
  produces<RPCRecHitCollection>("RPCTrackExtrapolatedPoints");
}

void RPCPointProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  if(incldt){
    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    iEvent.getByToken(dt4DSegments, all4DSegments);
    if(all4DSegments.isValid()){
      DTSegtoRPC DTClass(all4DSegments.product(), iSetup, debug, ExtrapolatedRegion);
      iEvent.put(DTClass.thePoints(), "RPCDTExtrapolatedPoints"); 
    }else{
      if(debug) std::cout<<"RPCHLT Invalid DTSegments collection"<<std::endl;
    }
  }

  if(inclcsc){
    edm::Handle<CSCSegmentCollection> allCSCSegments;
    iEvent.getByToken(cscSegments, allCSCSegments);
    if(allCSCSegments.isValid()){
      CSCSegtoRPC CSCClass(allCSCSegments.product(), iSetup, debug, ExtrapolatedRegion);
      iEvent.put(CSCClass.thePoints(), "RPCCSCExtrapolatedPoints"); 
    }else{
      if(debug) std::cout<<"RPCHLT Invalid CSCSegments collection"<<std::endl;
    }
  }
  if(incltrack){
    edm::Handle<reco::TrackCollection> alltracks;
    iEvent.getByToken(tracks,alltracks);
    if(!(alltracks->empty())){
      TracktoRPC TrackClass(alltracks.product(), iSetup, debug, trackTransformerParam, tracks_);
      iEvent.put(TrackClass.thePoints(), "RPCTrackExtrapolatedPoints");
    }else{
      if(debug) std::cout<<"RPCHLT Invalid Tracks collection"<<std::endl;
    }
  }
 
}
