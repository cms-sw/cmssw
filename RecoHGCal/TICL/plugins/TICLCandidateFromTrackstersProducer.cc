// user include files
#include <algorithm>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

using namespace ticl;

class TICLCandidateFromTrackstersProducer : public edm::stream::EDProducer<> {
public:
  TICLCandidateFromTrackstersProducer(const edm::ParameterSet&);
  ~TICLCandidateFromTrackstersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  std::vector<edm::EDGetTokenT<std::vector<Trackster>>> trackster_tokens_;
  // std::unique_ptr<PatternRecognitionAlgoBase> myAlgo_;
};
DEFINE_FWK_MODULE(TICLCandidateFromTrackstersProducer);

TICLCandidateFromTrackstersProducer::TICLCandidateFromTrackstersProducer(const edm::ParameterSet& ps) {
  trackster_tokens_ =
      edm::vector_transform(ps.getParameter<std::vector<edm::InputTag>>("tracksterCollections"),
                            [this](edm::InputTag const& tag) { return consumes<std::vector<Trackster>>(tag); });
  produces<std::vector<TICLCandidate>>();
}

void TICLCandidateFromTrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  std::vector<edm::InputTag> source_vector{edm::InputTag("TrackstersMIP"), edm::InputTag("Tracksters")};
  desc.add<std::vector<edm::InputTag>>("tracksterCollections", source_vector);
  descriptions.add("TICLCandidateFromTrackstersProducer", desc);
}

void TICLCandidateFromTrackstersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto result = std::make_unique<std::vector<TICLCandidate>>();
  auto output_mask = std::make_unique<std::vector<float>>();

  for (auto& trackster_token : trackster_tokens_) {
    edm::Handle<std::vector<Trackster>> trackster_h;
    evt.getByToken(trackster_token, trackster_h);

    // PRODUCE_CANDIDATE_FROM_TRACKSTER
    for (auto const& trackster : *trackster_h) {
      // PDG ID
      // auto max_index = std::max_element(trackster.id_probabilities.begin(),
      // trackster.id_probabilities.end());
      // auto pdg_id = pdg_id_from_idx(max_index);
    }
  }

  evt.put(std::move(result));
}
