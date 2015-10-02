// This producer assigns vertex times (with a specified resolution) to tracks.
// The times are produced as valuemaps associated to tracks, so the track dataformat doesn't
// need to be modified.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include <memory>


class TrackTimeValueMapProducer : public edm::EDProducer {
public:    
  TrackTimeValueMapProducer(const edm::ParameterSet&);
  ~TrackTimeValueMapProducer() { }
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  // inputs
  edm::EDGetTokenT<reco::TrackCollection> _tracks;
  edm::EDGetTokenT<reco::GsfTrackCollection> _gsfTracks;
  edm::EDGetTokenT<TrackingParticleCollection> _trackingParticles;
  edm::EDGetTokenT<TrackingVertexCollection> _trackingVertices;
  // options
};

DEFINE_FWK_MODULE(TrackTimeValueMapProducer);

namespace {
  static const std::string generalTracksName("generalTracks");
  static const std::string gsfTracksName("gsfTracks");
  static const std::string resolution("Resolution");
}

TrackTimeValueMapProducer::TrackTimeValueMapProducer(const edm::ParameterSet& conf) {
  // times and time resolutions for general tracks
  produces<edm::ValueMap<float> >(generalTracksName);
  produces<edm::ValueMap<float> >(generalTracksName+resolution);

  //for gsf tracks
  produces<edm::ValueMap<float> >(gsfTracksName);
  produces<edm::ValueMap<float> >(gsfTracksName+resolution);
}

void TrackTimeValueMapProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  std::auto_ptr<edm::ValueMap<float> > generalTrackTimes, generalTrackResolutions;
  std::auto_ptr<edm::ValueMap<float> > gsfTrackTimes, gsfTrackResolutions;

  generalTrackTimes.reset( new edm::ValueMap<float>() );
  generalTrackResolutions.reset( new edm::ValueMap<float>() );
  gsfTrackTimes.reset( new edm::ValueMap<float>() );
  gsfTrackResolutions.reset( new edm::ValueMap<float>() );

  evt.put( generalTrackTimes, generalTracksName );
  evt.put( generalTrackResolutions, generalTracksName+resolution );
  evt.put( gsfTrackTimes, gsfTracksName );
  evt.put( gsfTrackResolutions, gsfTracksName+resolution );
}
