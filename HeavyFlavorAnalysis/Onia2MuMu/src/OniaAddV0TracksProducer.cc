#include "HeavyFlavorAnalysis/Onia2MuMu/interface/OniaAddV0TracksProducer.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include <vector>

OniaAddV0TracksProducer::OniaAddV0TracksProducer(const edm::ParameterSet& ps)
    : events_v0{0}, total_v0{0}, total_lambda{0}, total_kshort{0} {
  LambdaCollectionToken_ =
      consumes<reco::VertexCompositeCandidateCollection>(ps.getParameter<edm::InputTag>("LambdaTag"));
  KShortCollectionToken_ =
      consumes<reco::VertexCompositeCandidateCollection>(ps.getParameter<edm::InputTag>("KShortTag"));

  produces<pat::CompositeCandidateCollection>("Kshort");
  produces<pat::CompositeCandidateCollection>("Lambda");
}

void OniaAddV0TracksProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& esetup) const {
  // Create unique_ptr for each collection to be stored in the Event
  std::unique_ptr<pat::CompositeCandidateCollection> Enhanced_kShortCandidates(new pat::CompositeCandidateCollection);
  std::unique_ptr<pat::CompositeCandidateCollection> Enhanced_lambdaCandidates(new pat::CompositeCandidateCollection);

  edm::Handle<reco::VertexCompositeCandidateCollection> lcandidates;
  event.getByToken(LambdaCollectionToken_, lcandidates);

  edm::Handle<reco::VertexCompositeCandidateCollection> kcandidates;
  event.getByToken(KShortCollectionToken_, kcandidates);

  int exits_l = 0;
  int exits_k = 0;
  for (reco::VertexCompositeCandidateCollection::const_iterator ik = kcandidates->begin(); ik != kcandidates->end();
       ++ik) {
    pat::CompositeCandidate kc{};
    edm::RefToBase<reco::Track> ktrk0((*(dynamic_cast<const reco::RecoChargedCandidate*>(ik->daughter(0)))).track());
    edm::RefToBase<reco::Track> ktrk1((*(dynamic_cast<const reco::RecoChargedCandidate*>(ik->daughter(1)))).track());
    kc.addUserData<reco::Track>("track0", *ktrk0);
    kc.addUserData<reco::Track>("track1", *ktrk1);
    Enhanced_kShortCandidates->emplace_back(std::move(kc));
    exits_k++;
  }

  for (reco::VertexCompositeCandidateCollection::const_iterator il = lcandidates->begin(); il != lcandidates->end();
       ++il) {
    pat::CompositeCandidate lc{};
    edm::RefToBase<reco::Track> ltrk0((*(dynamic_cast<const reco::RecoChargedCandidate*>(il->daughter(0)))).track());
    edm::RefToBase<reco::Track> ltrk1((*(dynamic_cast<const reco::RecoChargedCandidate*>(il->daughter(1)))).track());
    lc.addUserData<reco::Track>("track0", *ltrk0);
    lc.addUserData<reco::Track>("track1", *ltrk1);
    Enhanced_lambdaCandidates->emplace_back(std::move(lc));
    exits_l++;
  }

  // Write the collections to the Event

  total_v0 += exits_k;
  total_v0 += exits_l;
  total_kshort += exits_k;
  total_lambda += exits_l;
  if (exits_k || exits_l)
    events_v0++;

  event.put(std::move(Enhanced_kShortCandidates), "Kshort");
  event.put(std::move(Enhanced_lambdaCandidates), "Lambda");
}

void OniaAddV0TracksProducer::endJob() {
  edm::LogVerbatim("OniaAddV0TracksSummary") << "############################\n"
                                                "OniaAddV0Tracks producer report \n"
                                                "############################\n"
                                                "Total events with v0 :      "
                                             << events_v0
                                             << "\n"
                                                "Total v0 :                  "
                                             << total_v0
                                             << "\n"
                                                "Total number of lambda :    "
                                             << total_lambda
                                             << "\n"
                                                "Total number of kshort :    "
                                             << total_kshort
                                             << "\n"
                                                "############################";
}
//define this as a plug-in
DEFINE_FWK_MODULE(OniaAddV0TracksProducer);
