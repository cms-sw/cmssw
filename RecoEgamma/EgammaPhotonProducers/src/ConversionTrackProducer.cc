//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           ConversionTrackProducer
//
// Description:     Trivial producer of ConversionTrack collection from an edm::View of a track collection
//                  (ConversionTrack is a simple wrappper class containing a TrackBaseRef and some additional flags)
//
// Original Author: J.Bendavid
//
//

#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackProducer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

ConversionTrackProducer::ConversionTrackProducer(edm::ParameterSet const& conf)
    : useTrajectory(conf.getParameter<bool>("useTrajectory")),
      setTrackerOnly(conf.getParameter<bool>("setTrackerOnly")),
      setIsGsfTrackOpen(conf.getParameter<bool>("setIsGsfTrackOpen")),
      setArbitratedEcalSeeded(conf.getParameter<bool>("setArbitratedEcalSeeded")),
      setArbitratedMerged(conf.getParameter<bool>("setArbitratedMerged")),
      setArbitratedMergedEcalGeneral(conf.getParameter<bool>("setArbitratedMergedEcalGeneral")),
      beamSpotInputTag(consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotInputTag"))),
      filterOnConvTrackHyp(conf.getParameter<bool>("filterOnConvTrackHyp")),
      minConvRadius(conf.getParameter<double>("minConvRadius")) {
  edm::InputTag thetp(conf.getParameter<std::string>("TrackProducer"));
  genericTracks = consumes<edm::View<reco::Track> >(thetp);
  if (useTrajectory) {
    kfTrajectories = consumes<TrajTrackAssociationCollection>(thetp);
    gsfTrajectories = consumes<TrajGsfTrackAssociationCollection>(thetp);
  }
  magFieldToken = esConsumes();
  produces<reco::ConversionTrackCollection>();
}

// Virtual destructor needed.
ConversionTrackProducer::~ConversionTrackProducer() {}

// Functions that gets called by framework every event
void ConversionTrackProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  //get input collection (through edm::View)
  edm::View<reco::Track> const& trks = e.get(genericTracks);

  //get association maps between trajectories and tracks and build temporary maps
  std::map<reco::TrackRef, edm::Ref<std::vector<Trajectory> > > tracktrajmap;
  std::map<reco::GsfTrackRef, edm::Ref<std::vector<Trajectory> > > gsftracktrajmap;

  if (useTrajectory) {
    if (!trks.empty()) {
      if (dynamic_cast<const reco::GsfTrack*>(&trks.at(0))) {
        //fill map for gsf tracks
        for (auto const& pair : e.get(gsfTrajectories)) {
          gsftracktrajmap[pair.val] = pair.key;
        }
      } else {
        //fill map for standard tracks
        for (auto const& pair : e.get(kfTrajectories)) {
          tracktrajmap[pair.val] = pair.key;
        }
      }
    }
  }

  // Step B: create empty output collection
  auto outputTrks = std::make_unique<reco::ConversionTrackCollection>();

  //--------------------------------------------------
  //Added by D. Giordano
  // 2011/08/05
  // Reduction of the track sample based on geometric hypothesis for conversion tracks

  math::XYZVector beamSpot{e.get(beamSpotInputTag).position()};

  ConvTrackPreSelector.setMagnField(&es.getData(magFieldToken));

  //----------------------------------------------------------

  // Simple conversion of tracks to conversion tracks, setting appropriate flags from configuration
  for (size_t i = 0; i < trks.size(); ++i) {
    //--------------------------------------------------
    //Added by D. Giordano
    // 2011/08/05
    // Reduction of the track sample based on geometric hypothesis for conversion tracks

    edm::RefToBase<reco::Track> trackBaseRef = trks.refAt(i);
    if (filterOnConvTrackHyp &&
        ConvTrackPreSelector.isTangentPointDistanceLessThan(minConvRadius, trackBaseRef.get(), beamSpot))
      continue;
    //--------------------------------------------------

    reco::ConversionTrack convTrack(trackBaseRef);
    convTrack.setIsTrackerOnly(setTrackerOnly);
    convTrack.setIsGsfTrackOpen(setIsGsfTrackOpen);
    convTrack.setIsArbitratedEcalSeeded(setArbitratedEcalSeeded);
    convTrack.setIsArbitratedMerged(setArbitratedMerged);
    convTrack.setIsArbitratedMergedEcalGeneral(setArbitratedMergedEcalGeneral);

    //fill trajectory association if configured, using correct map depending on track type
    if (useTrajectory) {
      if (!gsftracktrajmap.empty()) {
        convTrack.setTrajRef(gsftracktrajmap.find(trackBaseRef.castTo<reco::GsfTrackRef>())->second);
      } else {
        convTrack.setTrajRef(tracktrajmap.find(trackBaseRef.castTo<reco::TrackRef>())->second);
      }
    }

    outputTrks->push_back(convTrack);
  }

  e.put(std::move(outputTrks));
  return;

}  //end produce
