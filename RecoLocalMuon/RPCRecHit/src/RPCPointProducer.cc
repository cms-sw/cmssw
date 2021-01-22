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

RPCPointProducer::RPCPointProducer(const edm::ParameterSet& iConfig)
    : incldt(iConfig.getUntrackedParameter<bool>("incldt", true)),
      inclcsc(iConfig.getUntrackedParameter<bool>("inclcsc", true)),
      incltrack(iConfig.getUntrackedParameter<bool>("incltrack", true)),
      debug(iConfig.getUntrackedParameter<bool>("debug", false)),
      MinCosAng(iConfig.getUntrackedParameter<double>("MinCosAng", 0.95)),
      MaxD(iConfig.getUntrackedParameter<double>("MaxD", 80.)),
      MaxDrb4(iConfig.getUntrackedParameter<double>("MaxDrb4", 150.)),
      ExtrapolatedRegion(iConfig.getUntrackedParameter<double>("ExtrapolatedRegion", 0.5)) {
  if (incldt) {
    dt4DSegments = consumes<DTRecSegment4DCollection>(iConfig.getParameter<edm::InputTag>("dt4DSegments"));
    dtSegtoRPC = std::make_unique<DTSegtoRPC>(consumesCollector());
  }
  if (inclcsc) {
    cscSegments = consumes<CSCSegmentCollection>(iConfig.getParameter<edm::InputTag>("cscSegments"));
    cscSegtoRPC = std::make_unique<CSCSegtoRPC>(consumesCollector());
  }
  if (incltrack) {
    tracks = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
    tracktoRPC = std::make_unique<TracktoRPC>(iConfig.getParameter<edm::ParameterSet>("TrackTransformer"),
                                              iConfig.getParameter<edm::InputTag>("tracks"),
                                              consumesCollector());
  }

  produces<RPCRecHitCollection>("RPCDTExtrapolatedPoints");
  produces<RPCRecHitCollection>("RPCCSCExtrapolatedPoints");
  produces<RPCRecHitCollection>("RPCTrackExtrapolatedPoints");
}

void RPCPointProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (incldt) {
    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    iEvent.getByToken(dt4DSegments, all4DSegments);
    if (all4DSegments.isValid()) {
      iEvent.put(dtSegtoRPC->thePoints(all4DSegments.product(), iSetup, debug, ExtrapolatedRegion),
                 "RPCDTExtrapolatedPoints");
    } else {
      if (debug)
        std::cout << "RPCHLT Invalid DTSegments collection" << std::endl;
    }
  }

  if (inclcsc) {
    edm::Handle<CSCSegmentCollection> allCSCSegments;
    iEvent.getByToken(cscSegments, allCSCSegments);
    if (allCSCSegments.isValid()) {
      iEvent.put(cscSegtoRPC->thePoints(allCSCSegments.product(), iSetup, debug, ExtrapolatedRegion),
                 "RPCCSCExtrapolatedPoints");
    } else {
      if (debug)
        std::cout << "RPCHLT Invalid CSCSegments collection" << std::endl;
    }
  }
  if (incltrack) {
    edm::Handle<reco::TrackCollection> alltracks;
    iEvent.getByToken(tracks, alltracks);
    if (!(alltracks->empty())) {
      iEvent.put(tracktoRPC->thePoints(alltracks.product(), iSetup, debug), "RPCTrackExtrapolatedPoints");
    } else {
      if (debug)
        std::cout << "RPCHLT Invalid Tracks collection" << std::endl;
    }
  }
}
