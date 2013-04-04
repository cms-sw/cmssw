#include "RecoMET/METProducers/interface/TrackMETProducer.h"

TrackMETProducer::TrackMETProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  src_ = cfg.getParameter<edm::InputTag>("src");
  globalThreshold_ = cfg.getParameter<double>("globalThreshold");

  produces<reco::METCollection>();
}

void TrackMETProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  double metPx = 0.;
  double metPy = 0.;
  double sumEt = 0.;
  
  edm::Handle<reco::TrackCollection> tracks;
  evt.getByLabel(src_, tracks);
  
  for ( reco::TrackCollection::const_iterator track = tracks->begin();
	track != tracks->end(); ++track ) {
    metPx -= track->px();
    metPy -= track->py();
    sumEt += track->pt();
  }
  
  double metPt = sqrt(metPx*metPx + metPy*metPy);
  
  reco::Candidate::LorentzVector trackMEtP4(metPx, metPy, 0., metPt);
  reco::MET trackMEt(sumEt, trackMEtP4, reco::Candidate::Point(0.,0.,0.));
  
  std::auto_ptr<reco::METCollection> trackMEtCollection(new reco::METCollection());
  trackMEtCollection->push_back(trackMEt);
  evt.put(trackMEtCollection);
}




 
