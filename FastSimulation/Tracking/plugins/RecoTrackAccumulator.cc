#include "FastSimulation/Tracking/plugins/RecoTrackAccumulator.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"


RecoTrackAccumulator::RecoTrackAccumulator(const edm::ParameterSet& conf, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
  signalTracksTag(conf.getParameter<edm::InputTag>("signalTracks")),
  signalMVAValuesTag(conf.getParameter<edm::InputTag>("signalMVAValues")),
  pileUpTracksTag(conf.getParameter<edm::InputTag>("pileUpTracks")),
  pileUpMVAValuesTag(conf.getParameter<edm::InputTag>("pileUpMVAValues")),
  outputLabel(conf.getParameter<std::string>("outputLabel")),
  MVAOutputLabel(conf.getParameter<std::string>("MVAOutputLabel"))
{
  mixMod.produces<reco::TrackCollection>(outputLabel);
  mixMod.produces<TrackingRecHitCollection>(outputLabel);
  mixMod.produces<reco::TrackExtraCollection>(outputLabel); 
  mixMod.produces<edm::ValueMap<float> >(MVAOutputLabel);
  
  iC.consumes<reco::TrackCollection>(signalTracksTag);
  iC.consumes<TrackingRecHitCollection>(signalTracksTag);
  iC.consumes<reco::TrackExtraCollection>(signalTracksTag);
  iC.consumes<edm::ValueMap<float> >(signalMVAValuesTag);
}

RecoTrackAccumulator::~RecoTrackAccumulator() {
    
}
  
void RecoTrackAccumulator::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
    
  newTracks_ = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection);
  newHits_ = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection);
  newTrackExtras_ = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection);
  newMVAVals_.clear();
  
  // this is needed to get the ProductId of the TrackExtra and TrackingRecHit and Track collections
  rNewTracks=const_cast<edm::Event&>( e ).getRefBeforePut<reco::TrackCollection>(outputLabel);
  rNewTrackExtras=const_cast<edm::Event&>( e ).getRefBeforePut<reco::TrackExtraCollection>(outputLabel);
  rNewHits=const_cast<edm::Event&>( e ).getRefBeforePut<TrackingRecHitCollection>(outputLabel);
}
  
void RecoTrackAccumulator::accumulate(edm::Event const& e, edm::EventSetup const& iSetup) {
  accumulateEvent( e, iSetup,signalTracksTag,signalMVAValuesTag);
}

void RecoTrackAccumulator::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& iSetup, edm::StreamID const&) {
  if (e.bunchCrossing()==0) {
    accumulateEvent( e, iSetup,pileUpTracksTag,pileUpMVAValuesTag);
  }
}

void RecoTrackAccumulator::finalizeEvent(edm::Event& e, const edm::EventSetup& iSetup) {
  
  std::auto_ptr< edm::ValueMap<float> > _newMVAVals(new edm::ValueMap<float>);
  edm::ValueMap<float>::Filler filler(*_newMVAVals);
  edm::TestHandle<reco::TrackCollection> newTracksHandle(newTracks_.get(),rNewTracks.id());
  filler.insert(newTracksHandle,newMVAVals_.begin(),newMVAVals_.end());
  filler.fill();
  e.put( newTracks_, outputLabel );
  e.put( newHits_, outputLabel );
  e.put( newTrackExtras_, outputLabel );
  e.put(_newMVAVals,MVAOutputLabel);
}


template<class T> void RecoTrackAccumulator::accumulateEvent(const T& e, edm::EventSetup const& iSetup,const edm::InputTag & label,const edm::InputTag & MVALabel) {

  edm::Handle<reco::TrackCollection> tracks;
  edm::Handle<TrackingRecHitCollection> hits;
  edm::Handle<reco::TrackExtraCollection> trackExtras;
  edm::Handle<edm::ValueMap<float> > mvaVals;
  if(!(e.getByLabel(label, tracks) and e.getByLabel(label, hits) and e.getByLabel(label, trackExtras))){
    edm::LogError ("RecoTrackAccumulator") << "Failed to find track, hit or trackExtra collections with inputTag " << label;
    exit(1);
  }
  else if(!e.getByLabel(MVALabel,mvaVals)){
    edm::LogError ("RecoTrackAccumulator") << "Failed to find mva values with inputTag" << MVALabel;
    exit(1);
  }

  if(tracks->size() != mvaVals->size()){
    edm::LogError("RecoTrackAccumulator") << "RecoTrackAccumulator expects the track collection and the MVA values to have the same number of entries" << std::endl;
  }
  if (tracks->size()==0)
    return;

  // crude way to get the track product id
  // (for PU events, the usual reco::TrackRef(tracks,t) does not result in a proper product id in the TrackRef)
  edm::ProductID tracksProdId = mvaVals->begin().id();
  
  for (size_t t = 0; t < tracks->size();++t){
    const reco::Track & track = tracks->at(t);
    newTracks_->push_back(track);
    // track extras:
    auto const& extra = trackExtras->at(track.extra().key());
    newTrackExtras_->emplace_back(extra.outerPosition(), extra.outerMomentum(), extra.outerOk(),
				  extra.innerPosition(),extra.innerMomentum(), extra.innerOk(),
				  extra.outerStateCovariance(), extra.outerDetId(),
				  extra.innerStateCovariance(), extra.innerDetId(),
				  extra.seedDirection(),
				  //If TrajectorySeeds are needed, then their list must be gotten from the
				  // secondary event directly and looked up similarly to TrackExtras.
				  //We can't use a default constructed RefToBase due to a bug in RefToBase
				  // which causes an seg fault when calling isAvailable on a default constructed one.
				  edm::RefToBase<TrajectorySeed>{edm::Ref<std::vector<TrajectorySeed>>{}});
    newTracks_->back().setExtra( reco::TrackExtraRef( rNewTrackExtras, newTracks_->size() - 1) );
    // rechits:
    // note: extra.recHit(i) does not work for pileup events
    // probably the Ref does not know its product id applies on a pileup event
    auto & newExtra = newTrackExtras_->back();
    auto const firstTrackIndex = newHits_->size();
    for(unsigned int i = 0;i<extra.recHitsSize();i++){
      newHits_->push_back( (*hits)[extra.recHit(i).key()] );
    }
    newExtra.setHits( rNewHits, firstTrackIndex, newHits_->size() - firstTrackIndex);
    // crude way to get the mva value that belongs to the track
    // the usual way valueMap[trackref] does not work in PU events, 
    // see earlier comment about track id
    float _val = mvaVals->get(tracksProdId,t);
    newMVAVals_.push_back(_val);
  }
}
