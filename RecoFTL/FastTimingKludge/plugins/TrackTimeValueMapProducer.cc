// This producer assigns vertex times (with a specified resolution) to tracks.
// The times are produced as valuemaps associated to tracks, so the track dataformat doesn't
// need to be modified.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

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
  edm::EDGetTokenT<edm::View<reco::Track> > _tracks;
  edm::EDGetTokenT<edm::View<reco::GsfTrack> > _gsfTracks;
  edm::EDGetTokenT<TrackingParticleCollection> _trackingParticles;
  edm::EDGetTokenT<TrackingVertexCollection> _trackingVertices;
  // options
};

DEFINE_FWK_MODULE(TrackTimeValueMapProducer);

namespace {
  static const std::string generalTracksName("generalTracks");
  static const std::string gsfTracksName("gsfTracks");
  static const std::string resolution("Resolution");

  template<typename ParticleType, typename T>
  void writeValueMap(edm::Event &iEvent,
                     const edm::Handle<edm::View<ParticleType> > & handle,
                     const std::vector<T> & values,
                     const std::string    & label) {
    std::auto_ptr<edm::ValueMap<T> > valMap(new edm::ValueMap<T>());
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap, label);
  }
}

TrackTimeValueMapProducer::TrackTimeValueMapProducer(const edm::ParameterSet& conf) :
  _tracks(consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("trackSrc") ) ),
  _gsfTracks(consumes<edm::View<reco::GsfTrack> >( conf.getParameter<edm::InputTag>("gsfTrackSrc") ) ),
  _trackingParticles(consumes<TrackingParticleCollection>( conf.getParameter<edm::InputTag>("trackingParticleSrc") ) ),
  _trackingVertices(consumes<TrackingVertexCollection>( conf.getParameter<edm::InputTag>("trackingVertexSrc") ) )
{
  // times and time resolutions for general tracks
  produces<edm::ValueMap<float> >(generalTracksName);
  produces<edm::ValueMap<float> >(generalTracksName+resolution);

  //for gsf tracks
  produces<edm::ValueMap<float> >(gsfTracksName);
  produces<edm::ValueMap<float> >(gsfTracksName+resolution);
}

void TrackTimeValueMapProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  std::vector<float> generalTrackTimes, generalTrackResolutions;
  std::vector<float> gsfTrackTimes, gsfTrackResolutions;
  
  edm::Handle<edm::View<reco::Track> > TrackCollectionH;
  evt.getByToken(_tracks, TrackCollectionH);
  const edm::View<reco::Track>& TrackCollection = *TrackCollectionH;

  edm::Handle<edm::View<reco::GsfTrack> > GsfTrackCollectionH;
  evt.getByToken(_gsfTracks, GsfTrackCollectionH);
  const edm::View<reco::GsfTrack>& GsfTrackCollection = *GsfTrackCollectionH;

  edm::Handle<TrackingParticleCollection>  TPCollectionH;
  evt.getByToken(_trackingParticles, TPCollectionH);
  const TrackingParticleCollection&  TPCollection = *TPCollectionH;

  edm::Handle<TrackingVertexCollection>  TVCollectionH;
  evt.getByToken(_trackingVertices, TVCollectionH);
  const TrackingVertexCollection&  TVCollection= *TVCollectionH;

  std::cout << "Track size            = " << TrackCollection.size() << std::endl;
  std::cout << "GsfTrack size         = " << GsfTrackCollection.size() << std::endl;
  std::cout << "TrackingParticle size = " << TPCollection.size() << std::endl;
  std::cout << "TrackingVertex size   = " << TVCollection.size() << std::endl;

  writeValueMap( evt, TrackCollectionH, generalTrackTimes, generalTracksName );
  writeValueMap( evt, TrackCollectionH, generalTrackResolutions, generalTracksName+resolution );
  writeValueMap( evt, GsfTrackCollectionH, gsfTrackTimes, gsfTracksName );
  writeValueMap( evt, GsfTrackCollectionH, gsfTrackResolutions, gsfTracksName+resolution );
}
