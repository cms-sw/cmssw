#include "RecoTrackAccumulator.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

RecoTrackAccumulator::RecoTrackAccumulator(const edm::ParameterSet& conf, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
  GeneralTrackInputSignal_(conf.getParameter<edm::InputTag>("GeneralTrackInputSignal")),
  GeneralTrackInputPileup_(conf.getParameter<edm::InputTag>("GeneralTrackInputPileup")),
  GeneralTrackOutput_(conf.getParameter<std::string>("GeneralTrackOutput")),
  HitInputSignal_(conf.getParameter<edm::InputTag>("HitInputSignal")),
  HitInputPileup_(conf.getParameter<edm::InputTag>("HitInputPileup")),
  HitOutput_(conf.getParameter<std::string>("HitOutput")),
  GeneralTrackExtraInputSignal_(conf.getParameter<edm::InputTag>("GeneralTrackExtraInputSignal")),
  GeneralTrackExtraInputPileup_(conf.getParameter<edm::InputTag>("GeneralTrackExtraInputPileup")),
  GeneralTrackExtraOutput_(conf.getParameter<std::string>("GeneralTrackExtraOutput"))
{

  mixMod.produces<reco::TrackCollection>(GeneralTrackOutput_);
  mixMod.produces<TrackingRecHitCollection>(HitOutput_);
  mixMod.produces<reco::TrackExtraCollection>(GeneralTrackExtraOutput_);

  iC.consumes<reco::TrackCollection>(GeneralTrackInputSignal_);
  iC.consumes<TrackingRecHitCollection>(HitInputSignal_);
  iC.consumes<reco::TrackExtraCollection>(GeneralTrackExtraInputSignal_);

  iC.consumes<reco::TrackCollection>(GeneralTrackInputPileup_);
  iC.consumes<TrackingRecHitCollection>(HitInputPileup_);
  iC.consumes<reco::TrackExtraCollection>(GeneralTrackExtraInputPileup_);
}
  
RecoTrackAccumulator::~RecoTrackAccumulator() {
    
}
  
void RecoTrackAccumulator::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
    
  NewTrackList_ = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection());
  NewHitList_ = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
  NewTrackExtraList_ = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());

  // this is needed to get the ProductId of the TrackExtra and TrackingRecHit collections
  rTrackExtras=const_cast<edm::Event&>( e ).getRefBeforePut<reco::TrackExtraCollection>(GeneralTrackExtraOutput_);
  rHits=const_cast<edm::Event&>( e ).getRefBeforePut<TrackingRecHitCollection>(HitOutput_);

}
  
void RecoTrackAccumulator::accumulate(edm::Event const& e, edm::EventSetup const& iSetup) {
  

  edm::Handle<reco::TrackCollection> tracks;
  edm::Handle<TrackingRecHitCollection> hits;
  edm::Handle<reco::TrackExtraCollection> trackExtras;
  e.getByLabel(GeneralTrackInputSignal_, tracks);
  e.getByLabel(HitInputSignal_, hits);
  e.getByLabel(GeneralTrackExtraInputSignal_, trackExtras);

  // Call the templated version that does the same for both signal and pileup events
  accumulateEvent( e, iSetup, tracks, trackExtras, hits );

}


void RecoTrackAccumulator::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& iSetup) {

  if (e.bunchCrossing()==0) {
    edm::Handle<reco::TrackCollection> tracks;
    edm::Handle<TrackingRecHitCollection> hits;
    edm::Handle<reco::TrackExtraCollection> trackExtras;
    e.getByLabel(GeneralTrackInputPileup_, tracks);
    e.getByLabel(HitInputPileup_, hits);
    e.getByLabel(GeneralTrackExtraInputPileup_, trackExtras);
    
    // Call the templated version that does the same for both signal and pileup events
    accumulateEvent( e, iSetup, tracks, trackExtras, hits );

  }
}

void RecoTrackAccumulator::finalizeEvent(edm::Event& e, const edm::EventSetup& iSetup) {
  
  e.put( NewTrackList_, GeneralTrackOutput_ );
  e.put( NewTrackExtraList_, GeneralTrackExtraOutput_ );

}


template<class T> void RecoTrackAccumulator::accumulateEvent(const T& e, edm::EventSetup const& iSetup, edm::Handle<reco::TrackCollection> tracks, edm::Handle<reco::TrackExtraCollection> tracksExtras, edm::Handle<TrackingRecHitCollection> hits) {

  if (tracks.isValid()) {
    short track_counter = 0;
    //    short hit_counter = 0;
    for (auto const& track : *tracks) {
      NewTrackList_->push_back(track);
      // track extras:
      NewTrackExtraList_->push_back(tracksExtras->at(track_counter));
      NewTrackList_->back().setExtra( reco::TrackExtraRef( rTrackExtras, NewTrackExtraList_->size() - 1) );
      //reco::TrackExtra & tx = NewTrackExtraList_->back();
      //tx.setResiduals(track.residuals());
      // rechits:
      for( trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++ hit ) {
	NewHitList_->push_back( (*hit)->clone() ); // is this safe?
	//	tx.add( TrackingRecHitRef( rHits, NewHitList_->size() - 1) );
	//	hit_counter++;
      }

      track_counter++;
    }
  }

}
