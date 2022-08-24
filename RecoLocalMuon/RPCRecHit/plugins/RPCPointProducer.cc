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

#include "RPCPointProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// system include files

#include <memory>
#include <ctime>

// user include files

void RPCPointProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("incldt", true);
  desc.add<bool>("inclcsc", true);
  desc.add<bool>("incltrack", false);
  desc.addUntracked<bool>("debug", false);
  desc.add<double>("rangestrips", 4.);
  desc.add<double>("rangestripsRB4", 4.);
  desc.add<double>("MinCosAng", 0.85);
  desc.add<double>("MaxD", 80.0);
  desc.add<double>("MaxDrb4", 150.0);
  desc.add<double>("ExtrapolatedRegion", 0.5);
  desc.add<edm::InputTag>("cscSegments", edm::InputTag("hltCscSegments"));
  desc.add<edm::InputTag>("dt4DSegments", edm::InputTag("hltDt4DSegments"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("standAloneMuons"));
  desc.add<int>("minBX", -2);
  desc.add<int>("maxBX", 2);
  edm::ParameterSetDescription descNested;
  descNested.add<bool>("DoPredictionsOnly", false);
  descNested.add<std::string>("Fitter", "KFFitterForRefitInsideOut");
  descNested.add<std::string>("TrackerRecHitBuilder", "WithTrackAngle");
  descNested.add<std::string>("Smoother", "KFSmootherForRefitInsideOut");
  descNested.add<std::string>("MuonRecHitBuilder", "MuonRecHitBuilder");
  descNested.add<std::string>("RefitDirection", "alongMomentum");
  descNested.add<bool>("RefitRPCHits", false);
  descNested.add<std::string>("Propagator", "SmartPropagatorAnyRKOpposite");
  desc.add<edm::ParameterSetDescription>("TrackTransformer", descNested);

  descriptions.add("rpcPointProducer", desc);
}

RPCPointProducer::RPCPointProducer(const edm::ParameterSet& iConfig)
    : incldt(iConfig.getParameter<bool>("incldt")),
      inclcsc(iConfig.getParameter<bool>("inclcsc")),
      incltrack(iConfig.getParameter<bool>("incltrack")),
      debug(iConfig.getUntrackedParameter<bool>("debug")),
      MinCosAng(iConfig.getParameter<double>("MinCosAng")),
      MaxD(iConfig.getParameter<double>("MaxD")),
      MaxDrb4(iConfig.getParameter<double>("MaxDrb4")),
      ExtrapolatedRegion(iConfig.getParameter<double>("ExtrapolatedRegion")) {
  if (incldt) {
    dt4DSegments = consumes<DTRecSegment4DCollection>(iConfig.getParameter<edm::InputTag>("dt4DSegments"));
    dtSegtoRPC = std::make_unique<DTSegtoRPC>(consumesCollector(), iConfig);
  }
  if (inclcsc) {
    cscSegments = consumes<CSCSegmentCollection>(iConfig.getParameter<edm::InputTag>("cscSegments"));
    cscSegtoRPC = std::make_unique<CSCSegtoRPC>(consumesCollector(), iConfig);
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
        LogDebug("RPCPointProducer") << "RPCHLT Invalid DTSegments collection" << std::endl;
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
        LogDebug("RPCPointProducer") << "RPCHLT Invalid CSCSegments collection" << std::endl;
    }
  }
  if (incltrack) {
    edm::Handle<reco::TrackCollection> alltracks;
    iEvent.getByToken(tracks, alltracks);
    if (!(alltracks->empty())) {
      iEvent.put(tracktoRPC->thePoints(alltracks.product(), iSetup, debug), "RPCTrackExtrapolatedPoints");
    } else {
      if (debug)
        LogDebug("RPCPointProducer") << "RPCHLT Invalid Tracks collection" << std::endl;
    }
  }
}
