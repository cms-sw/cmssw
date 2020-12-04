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

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaTrackReco/interface/ConversionTrack.h"
#include "DataFormats/EgammaTrackReco/interface/ConversionTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/ConversionSeedGenerators/interface/IdealHelixParameters.h"
#include "TrackingTools/GsfTracking/interface/TrajGsfTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include <string>
#include <vector>

class ConversionTrackProducer : public edm::stream::EDProducer<> {
  typedef edm::AssociationMap<edm::OneToOne<std::vector<Trajectory>, reco::GsfTrackCollection, unsigned short> >
      TrajGsfTrackAssociationCollection;

public:
  explicit ConversionTrackProducer(const edm::ParameterSet& conf);

  ~ConversionTrackProducer() override;

  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::EDGetTokenT<edm::View<reco::Track> > genericTracks;
  edm::EDGetTokenT<TrajTrackAssociationCollection> kfTrajectories;
  edm::EDGetTokenT<TrajGsfTrackAssociationCollection> gsfTrajectories;
  bool useTrajectory;
  bool setTrackerOnly;
  bool setIsGsfTrackOpen;
  bool setArbitratedEcalSeeded;
  bool setArbitratedMerged;
  bool setArbitratedMergedEcalGeneral;

  //--------------------------------------------------
  //Added by D. Giordano
  // 2011/08/05
  // Reduction of the track sample based on geometric hypothesis for conversion tracks

  edm::EDGetTokenT<reco::BeamSpot> beamSpotInputTag;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken;
  bool filterOnConvTrackHyp;
  double minConvRadius;
  IdealHelixParameters ConvTrackPreSelector;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ConversionTrackProducer);

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
