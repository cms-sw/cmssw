#include "HeavyFlavorAnalysis/Onia2MuMu/interface/OniaAddV0TracksProducer.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include <vector>

OniaAddV0TracksProducer:: OniaAddV0TracksProducer(const edm::ParameterSet& ps) {

  LambdaCollectionToken_ = consumes<reco::VertexCompositeCandidateCollection>(ps.getParameter<edm::InputTag>("LambdaTag"));
  KShortCollectionToken_ = consumes<reco::VertexCompositeCandidateCollection>(ps.getParameter<edm::InputTag>("KShortTag"));

  produces< pat::CompositeCandidateCollection >("Kshort");
  produces< pat::CompositeCandidateCollection >("Lambda");

  events_v0    = 0;
  total_v0     = 0;
  total_lambda = 0;
  total_kshort = 0;
  std::cout << "running OniaAddV0TracksProducer..." << std::endl;
}

void OniaAddV0TracksProducer::produce(edm::Event& event, const edm::EventSetup& esetup){

  // Create auto_ptr for each collection to be stored in the Event
  std::auto_ptr<pat::CompositeCandidateCollection> Enhanced_kShortCandidates(new pat::CompositeCandidateCollection);
  std::auto_ptr<pat::CompositeCandidateCollection> Enhanced_lambdaCandidates(new pat::CompositeCandidateCollection);

  edm::Handle<reco::VertexCompositeCandidateCollection> lcandidates;
  event.getByToken(LambdaCollectionToken_,lcandidates);

  edm::Handle<reco::VertexCompositeCandidateCollection> kcandidates;
  event.getByToken(KShortCollectionToken_,kcandidates);

  int exits_l = 0;
  int exits_k = 0;
  for (reco::VertexCompositeCandidateCollection::const_iterator ik = kcandidates->begin(); ik != kcandidates->end(); ++ik) {
      pat::CompositeCandidate *kc = new pat::CompositeCandidate();
      edm::RefToBase<reco::Track> ktrk0((*(dynamic_cast<const reco::RecoChargedCandidate*>(ik->daughter(0)))).track());
      edm::RefToBase<reco::Track> ktrk1((*(dynamic_cast<const reco::RecoChargedCandidate*>(ik->daughter(1)))).track());
      kc->addUserData<reco::Track>( "track0", *ktrk0 );
      kc->addUserData<reco::Track>( "track1", *ktrk1 );
      Enhanced_kShortCandidates->push_back(*kc);
      exits_k++;
  }

  for (reco::VertexCompositeCandidateCollection::const_iterator il = lcandidates->begin(); il != lcandidates->end(); ++il) {
      pat::CompositeCandidate *lc = new pat::CompositeCandidate();
      edm::RefToBase<reco::Track> ltrk0((*(dynamic_cast<const reco::RecoChargedCandidate*>(il->daughter(0)))).track());
      edm::RefToBase<reco::Track> ltrk1((*(dynamic_cast<const reco::RecoChargedCandidate*>(il->daughter(1)))).track());
      lc->addUserData<reco::Track>( "track0", *ltrk0 );
      lc->addUserData<reco::Track>( "track1", *ltrk1 );
      Enhanced_lambdaCandidates->push_back(*lc);
      exits_l++;
  }

  // Write the collections to the Event

  total_v0 += exits_k;
  total_v0 += exits_l;
  total_kshort += exits_k;
  total_lambda += exits_l;
  if (exits_k || exits_l) events_v0++;

  event.put( Enhanced_kShortCandidates,"Kshort");
  event.put( Enhanced_lambdaCandidates,"Lambda");
}

void OniaAddV0TracksProducer::endJob(){
   std::cout << "############################" << std::endl;
   std::cout << "OniaAddV0Tracks producer report " << std::endl;
   std::cout << "############################" << std::endl;
   std::cout << "Total events with v0 :      " << events_v0 << std::endl;
   std::cout << "Total v0 :                  " << total_v0 << std::endl;
   std::cout << "Total number of lambda :    " << total_lambda << std::endl;
   std::cout << "Total number of kshort :    " << total_kshort << std::endl;
   std::cout << "############################" << std::endl;
}
//define this as a plug-in
DEFINE_FWK_MODULE(OniaAddV0TracksProducer);
