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

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/QuickTrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include <memory>

#include "ResolutionModel.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "CLHEP/Random/RandGauss.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

class TrackTimeValueMapProducer : public edm::EDProducer {
public:    
  TrackTimeValueMapProducer(const edm::ParameterSet&);
  ~TrackTimeValueMapProducer() { }
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
private:
  // inputs
  const edm::EDGetTokenT<edm::View<reco::Track> > _tracks;
  const edm::EDGetTokenT<edm::View<reco::Track> > _gsfTracks;
  const edm::EDGetTokenT<TrackingParticleCollection> _trackingParticles;
  const edm::EDGetTokenT<TrackingVertexCollection> _trackingVertices;
  const edm::EDGetTokenT<edm::HepMCProduct> _hepMCProduct;
  // tp associator
  const std::string _associatorName;
  // options
  std::vector<std::unique_ptr<const ResolutionModel> > _resolutions;
  // functions
  void calculateTrackTimes( const edm::View<reco::Track>&, 
                            const reco::RecoToSimCollection&,
                            std::vector<float>& ) const;
  std::pair<float,float> extractTrackVertexTime(const std::vector<std::pair<TrackingParticleRef, double> >&) const;
  // RNG
  CLHEP::HepRandomEngine* _rng_engine;
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
  _gsfTracks(consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("gsfTrackSrc") ) ),
  _trackingParticles(consumes<TrackingParticleCollection>( conf.getParameter<edm::InputTag>("trackingParticleSrc") ) ),
  _trackingVertices(consumes<TrackingVertexCollection>( conf.getParameter<edm::InputTag>("trackingVertexSrc") ) ),
  _associatorName( conf.getParameter<std::string>("tpAssociator") )
{
  // setup resolution models
  const std::vector<edm::ParameterSet>& resos = conf.getParameterSetVector("resolutionModels");
  for( const auto& reso : resos ) {
    const std::string& name = reso.getParameter<std::string>("modelName");
    ResolutionModel* resomod = ResolutionModelFactory::get()->create(name,reso);
    _resolutions.emplace_back( resomod );  

    // times and time resolutions for general tracks
    produces<edm::ValueMap<float> >(generalTracksName+name);
    produces<edm::ValueMap<float> >(generalTracksName+name+resolution);
    
    //for gsf tracks
    produces<edm::ValueMap<float> >(gsfTracksName+name);
    produces<edm::ValueMap<float> >(gsfTracksName+name+resolution);
  }
  // get RNG engine
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()){
    throw cms::Exception("Configuration")
      << "TrackTimeValueMapProducer::TrackTimeValueMapProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
      << "Add the service in the configuration file or remove the modules that require it.";
  }
  _rng_engine = &(rng->getEngine());
}

void TrackTimeValueMapProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  std::vector<float> generalTrackTimes, gsfTrackTimes;

  //get associator
  edm::ESHandle<TrackAssociatorBase> theAssociator;
  es.get<TrackAssociatorRecord>().get(_associatorName,theAssociator);
  
  //get track collections
  edm::Handle<edm::View<reco::Track> > TrackCollectionH;
  evt.getByToken(_tracks, TrackCollectionH);
  const edm::View<reco::Track>& TrackCollection = *TrackCollectionH;

  edm::Handle<edm::View<reco::Track> > GsfTrackCollectionH;
  evt.getByToken(_gsfTracks, GsfTrackCollectionH);
  const edm::View<reco::Track>& GsfTrackCollection = *GsfTrackCollectionH;

  //get tracking particle collections
  edm::Handle<TrackingParticleCollection>  TPCollectionH;
  evt.getByToken(_trackingParticles, TPCollectionH);
  //const TrackingParticleCollection&  TPCollection = *TPCollectionH;

  /*
  edm::Handle<TrackingVertexCollection>  TVCollectionH;
  evt.getByToken(_trackingVertices, TVCollectionH);
  const TrackingVertexCollection&  TVCollection= *TVCollectionH;
  */

  /*
  std::cout << "Track size            = " << TrackCollection.size() << std::endl;
  std::cout << "GsfTrack size         = " << GsfTrackCollection.size() << std::endl;
  std::cout << "TrackingParticle size = " << TPCollection.size() << std::endl;
  std::cout << "TrackingVertex size   = " << TVCollection.size() << std::endl;
  */
  
  // associate the reco tracks / gsf Tracks
  reco::RecoToSimCollection generalRecSimColl, gsfRecSimColl;
  generalRecSimColl = theAssociator->associateRecoToSim(TrackCollectionH,
                                                        TPCollectionH,
                                                        &evt,&es);
  gsfRecSimColl = theAssociator->associateRecoToSim(GsfTrackCollectionH,
                                                    TPCollectionH,
                                                    &evt,&es);

  /*
  std::cout << "Track association sizes: " << generalRecSimColl.size() << ' ' 
            << gsfRecSimColl.size() << std::endl;
  */
  //std::cout << "tracksSrc" << std::endl;
  calculateTrackTimes(TrackCollection, generalRecSimColl, generalTrackTimes);
  //std::cout << "gsfTracksSrc" << std::endl;
  calculateTrackTimes(GsfTrackCollection, gsfRecSimColl, gsfTrackTimes);

  for( const auto& reso : _resolutions ) {
    const std::string& name = reso->name();
    std::vector<float> times, resos;
    std::vector<float> gsf_times, gsf_resos;
    
    times.reserve(TrackCollection.size());
    resos.reserve(TrackCollection.size());
    gsf_times.reserve(GsfTrackCollection.size());
    gsf_resos.reserve(GsfTrackCollection.size());

    for( unsigned i = 0; i < TrackCollection.size(); ++i ) {
      const reco::Track& tk = TrackCollection[i];
      if( edm::isFinite( generalTrackTimes[i] ) && generalTrackTimes[i] != 0.f) {
        const float resolution = reso->getTimeResolution(tk);
        times.push_back( CLHEP::RandGauss::shoot(_rng_engine, generalTrackTimes[i], resolution) );
        resos.push_back( resolution );
      } else {
        times.push_back( generalTrackTimes[i] );
        resos.push_back( 0.f );
      }
    }

    for( unsigned i = 0; i < GsfTrackCollection.size(); ++i ) {
      const reco::Track& tk = GsfTrackCollection[i];
      if( edm::isFinite( gsfTrackTimes[i] )  && gsfTrackTimes[i] != 0.f ) {
        const float resolution = reso->getTimeResolution(tk);
        gsf_times.push_back( CLHEP::RandGauss::shoot(_rng_engine, gsfTrackTimes[i], resolution) );
        gsf_resos.push_back( resolution ); 
      } else {
        gsf_times.push_back( gsfTrackTimes[i] );
        gsf_resos.push_back( 0.f );
      }
    }

    writeValueMap( evt, TrackCollectionH, times, generalTracksName+name );
    writeValueMap( evt, TrackCollectionH, resos, generalTracksName+name+resolution );
    writeValueMap( evt, GsfTrackCollectionH, gsf_times, gsfTracksName+name );
    writeValueMap( evt, GsfTrackCollectionH, gsf_resos, gsfTracksName+name+resolution );
  }
}

void TrackTimeValueMapProducer::calculateTrackTimes( const edm::View<reco::Track>& tkcoll,
                                                     const reco::RecoToSimCollection& assoc,
                                                     std::vector<float>& tvals ) const { 
  constexpr float flt_max = std::numeric_limits<float>::quiet_NaN();
  
  for( unsigned itk = 0; itk < tkcoll.size(); ++itk ) {
    const auto tkref = tkcoll.refAt(itk);
    auto track_tps = assoc.find(tkref);    
    if( track_tps != assoc.end() ) {
      if( !track_tps->val.size() ) {
        tvals.push_back(flt_max);
      } else {
        
        const std::pair<float,float> time_info = extractTrackVertexTime(track_tps->val);
        if( time_info.second*tkref->vz() < 0.f ) {
          std::cout << "track z = " << tkref->vz() << " +/- " << tkref->dzError() << " cm ,";
          std::cout << " sim vertex z = " << time_info.second << " t = "  << time_info.first;
          std::cout << std::endl;
        }
        tvals.push_back(time_info.first);
      }
    }
  } 
}

std::pair<float,float> TrackTimeValueMapProducer::
extractTrackVertexTime( const std::vector<std::pair<TrackingParticleRef, double> >& tp_list ) const {
  float result = 0.f;
  float result_z = 0.f;
  for( const auto& tpref : tp_list ) {
    const auto& tvertex = tpref.first->parentVertex();
    result = tvertex->position().T()*CLHEP::second; // convert into nano-seconds
    result_z = tvertex->position().Z();
    // account for secondary vertices...
    
    if( tvertex->nSourceTracks() ) {
      auto pvertex = tvertex->sourceTracks()[0]->parentVertex();
      result = pvertex->position().T()*CLHEP::second;
      result_z = pvertex->position().Z();
      while( pvertex->nSourceTracks() ) {
        pvertex = pvertex->sourceTracks()[0]->parentVertex();
        result = pvertex->position().T()*CLHEP::second;
        result_z = pvertex->position().Z();
      }
    }    
  }
  if( tp_list.size() > 1 ) LogDebug("TooManyTracks") << "track matched to " << tp_list.size() << " tracking particles!" << std::endl;
  return std::make_pair(result,result_z);
}
