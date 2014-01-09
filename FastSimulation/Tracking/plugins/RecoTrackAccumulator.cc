#include "RecoTrackAccumulator.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

RecoTrackAccumulator::RecoTrackAccumulator(const edm::ParameterSet& conf, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
  GeneralTrackInputSignal_(conf.getParameter<edm::InputTag>("GeneralTrackInputSignal")),
  GeneralTrackInputPileup_(conf.getParameter<edm::InputTag>("GeneralTrackInputPileup")),
  GeneralTrackOutput_(conf.getParameter<std::string>("GeneralTrackOutput")),
  GeneralTrackExtraInputSignal_(conf.getParameter<edm::InputTag>("GeneralTrackExtraInputSignal")),
  GeneralTrackExtraInputPileup_(conf.getParameter<edm::InputTag>("GeneralTrackExtraInputPileup")),
  GeneralTrackExtraOutput_(conf.getParameter<std::string>("GeneralTrackExtraOutput"))
{

  mixMod.produces<reco::TrackCollection>(GeneralTrackOutput_);
  mixMod.produces<reco::TrackExtraCollection>(GeneralTrackExtraOutput_);

  iC.consumes<reco::TrackCollection>(GeneralTrackInputSignal_);
  iC.consumes<reco::TrackExtraCollection>(GeneralTrackExtraInputSignal_);

  iC.consumes<reco::TrackCollection>(GeneralTrackInputPileup_);
  iC.consumes<reco::TrackExtraCollection>(GeneralTrackExtraInputPileup_);
}
  
RecoTrackAccumulator::~RecoTrackAccumulator() {
    
}
  
void RecoTrackAccumulator::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
    
  NewTrackList_ = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection());
  NewTrackExtraList_ = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());

  // this is needed to get the ProductId of the TrackExtra collection
  rTrackExtras=const_cast<edm::Event&>( e ).getRefBeforePut<reco::TrackExtraCollection>(GeneralTrackExtraOutput_);

}
  
void RecoTrackAccumulator::accumulate(edm::Event const& e, edm::EventSetup const& iSetup) {
  

  edm::Handle<reco::TrackCollection> tracks;
  edm::Handle<reco::TrackExtraCollection> trackExtras;
  e.getByLabel(GeneralTrackInputSignal_, tracks);
  e.getByLabel(GeneralTrackExtraInputSignal_, trackExtras);

  // Call the templated version that does the same for both signal and pileup events
  accumulateEvent( e, iSetup, tracks, trackExtras );

}


void RecoTrackAccumulator::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& iSetup) {

  if (e.bunchCrossing()==0) {
    edm::Handle<reco::TrackCollection> tracks;
    edm::Handle<reco::TrackExtraCollection> trackExtras;
    e.getByLabel(GeneralTrackInputPileup_, tracks);
    e.getByLabel(GeneralTrackExtraInputPileup_, trackExtras);
    
    // Call the templated version that does the same for both signal and pileup events
    accumulateEvent( e, iSetup, tracks, trackExtras );

  }
}

void RecoTrackAccumulator::finalizeEvent(edm::Event& e, const edm::EventSetup& iSetup) {
  
  e.put( NewTrackList_, GeneralTrackOutput_ );
  e.put( NewTrackExtraList_, GeneralTrackExtraOutput_ );

}


template<class T> void RecoTrackAccumulator::accumulateEvent(const T& e, edm::EventSetup const& iSetup, edm::Handle<reco::TrackCollection> tracks, edm::Handle<reco::TrackExtraCollection> tracksExtras) {

  if (tracks.isValid()) {
    short counter = 0;
    for (auto const& track : *tracks) {
      NewTrackList_->push_back(track);
      reco::TrackExtra tkExtra(tracksExtras->at(counter));
      NewTrackExtraList_->push_back(tkExtra);
      //      NewTrackList_->back().setExtra( reco::TrackExtraRef(tracksExtras, track.extra().key(), true) );
      NewTrackList_->back().setExtra( reco::TrackExtraRef( rTrackExtras, NewTrackExtraList_->size() - 1) );
      counter++;
    }
  }

}
