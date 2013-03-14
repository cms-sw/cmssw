#include "RecoTrackAccumulator.h"
#include "FWCore/Framework/interface/EDProducer.h"

RecoTrackAccumulator::RecoTrackAccumulator(const edm::ParameterSet& conf, edm::EDProducer& mixMod) {
    
  GeneralTrackInput_ = conf.getParameter<edm::InputTag>("GeneralTrackInput");
  GeneralTrackOutput_  = conf.getParameter<std::string>("GeneralTrackOutput");

  mixMod.produces<reco::TrackCollection>(GeneralTrackOutput_);
}
  
RecoTrackAccumulator::~RecoTrackAccumulator() {
    
}
  
void RecoTrackAccumulator::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
    
  NewTrackList_ = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection());

}
  
void RecoTrackAccumulator::accumulate(edm::Event const& e, edm::EventSetup const& iSetup) {
  

  edm::Handle<reco::TrackCollection> tracks;
  e.getByLabel(GeneralTrackInput_, tracks);
  
  if (tracks.isValid()) {
    for (reco::TrackCollection::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
      NewTrackList_->push_back(*track);
    }
  }

}

void RecoTrackAccumulator::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& iSetup) {

  
  edm::Handle<reco::TrackCollection> tracks;
  e.getByLabel(GeneralTrackInput_, tracks);
  
  if (tracks.isValid()) {
    for (reco::TrackCollection::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
      NewTrackList_->push_back(*track);
    }
  }

}

void RecoTrackAccumulator::finalizeEvent(edm::Event& e, const edm::EventSetup& iSetup) {
  
  e.put( NewTrackList_, GeneralTrackOutput_ );

}


