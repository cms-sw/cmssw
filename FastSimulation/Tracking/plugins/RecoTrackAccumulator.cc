#include "RecoTrackAccumulator.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

RecoTrackAccumulator::RecoTrackAccumulator(const edm::ParameterSet& conf, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
  Input_(conf.getParameter<edm::InputTag>("Input")),
  GeneralTrackOutput_(conf.getParameter<std::string>("GeneralTrackOutput")),
  HitOutput_(conf.getParameter<std::string>("HitOutput")),
  GeneralTrackExtraOutput_(conf.getParameter<std::string>("GeneralTrackExtraOutput"))
{

  mixMod.produces<reco::TrackCollection>(GeneralTrackOutput_);
  mixMod.produces<TrackingRecHitCollection>(HitOutput_);
  mixMod.produces<reco::TrackExtraCollection>(GeneralTrackExtraOutput_);

  iC.consumes<reco::TrackCollection>(Input_);
  iC.consumes<TrackingRecHitCollection>(Input_);
  iC.consumes<reco::TrackExtraCollection>(Input_);
}
  
RecoTrackAccumulator::~RecoTrackAccumulator() {
    
}
  
void RecoTrackAccumulator::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {

  std::cout << "RTA: init event..." << std::endl;    

  NewTrackList_ = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection());
  NewHitList_ = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
  NewTrackExtraList_ = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());

  // this is needed to get the ProductId of the TrackExtra and TrackingRecHit collections
  rTrackExtras=const_cast<edm::Event&>( e ).getRefBeforePut<reco::TrackExtraCollection>(GeneralTrackExtraOutput_);
  rHits=const_cast<edm::Event&>( e ).getRefBeforePut<TrackingRecHitCollection>(HitOutput_);

  std::cout << "RTA: ... done init event" << std::endl;    
}
  
void RecoTrackAccumulator::accumulate(edm::Event const& e, edm::EventSetup const& iSetup) {
    accumulateEvent( e, iSetup);
}

void RecoTrackAccumulator::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& iSetup, edm::StreamID const&) {
  if (e.bunchCrossing()==0) {
    accumulateEvent( e, iSetup);
  }
}

void RecoTrackAccumulator::finalizeEvent(edm::Event& e, const edm::EventSetup& iSetup) {
  
  e.put( NewTrackList_, GeneralTrackOutput_ );
  e.put( NewHitList_, HitOutput_ );
  e.put( NewTrackExtraList_, GeneralTrackExtraOutput_ );

}


template<class T> void RecoTrackAccumulator::accumulateEvent(const T& e, edm::EventSetup const& iSetup) {

  edm::Handle<reco::TrackCollection> tracks;
  edm::Handle<TrackingRecHitCollection> hits;
  edm::Handle<reco::TrackExtraCollection> trackExtras;
  std::cout << "RTA: get my stuff..." << std::endl;    
  if(!(e.getByLabel(Input_, tracks) and e.getByLabel(Input_, hits) and e.getByLabel(Input_, trackExtras))){
    edm::LogError ("Failed to find track, hit or trackExtra collections");
    exit(1);
  }
  std::cout << "RTA: got my stuff..." << std::endl;    

  for (auto const& track : *tracks) {
    NewTrackList_->push_back(track);
    // track extras:
    auto const& extra = trackExtras->at(track.extra().key());
    NewTrackExtraList_->emplace_back(extra.outerPosition(), extra.outerMomentum(), extra.outerOk(),
				     extra.innerPosition(),extra.innerMomentum(), extra.innerOk(),
				     extra.outerStateCovariance(), extra.outerDetId(),
				     extra.innerStateCovariance(), extra.innerDetId(),
				     extra.seedDirection(),
				     //If TrajectorySeeds are needed, then their list must be gotten from the
				     // secondary event directly and looked up similarly to TrackExtras.
				     //We can't use a default constructed RefToBase due to a bug in RefToBase
				     // which causes an seg fault when calling isAvailable on a default constructed one.
				     edm::RefToBase<TrajectorySeed>{edm::Ref<std::vector<TrajectorySeed>>{}});
    NewTrackList_->back().setExtra( reco::TrackExtraRef( rTrackExtras, NewTrackExtraList_->size() - 1) );
    // rechits:
    // note: extra.recHit(i) does not work for pileup events
    // probably the Ref does not know its product id applies on a pileup event
    auto & newExtra = NewTrackExtraList_->back();
    auto const firstTrackIndex = NewHitList_->size();
    for(unsigned int i = 0;i<extra.recHitsSize();i++){
      NewHitList_->push_back( (*hits)[extra.recHit(i).key()] );
    }
    newExtra.setHits( rHits, firstTrackIndex, NewHitList_->size() - firstTrackIndex);
  }

}
