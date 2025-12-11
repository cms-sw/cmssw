#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackTorchClassifierFeaturesSoA.h"

// This module consumes the HOST copy of the Alpaka device scores
// The framework automatically creates host copies of device PortableCollections
class TrackTorchClassifierFromSoA : public edm::stream::EDProducer<> {
public:
  explicit TrackTorchClassifierFromSoA(const edm::ParameterSet& iConfig);
  ~TrackTorchClassifierFromSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const edm::EDGetTokenT<reco::TrackCollection> tracks_token_;
  const edm::EDGetTokenT<PortableHostCollection<TrackTorchClassifierScoresSoA>> scores_token_;
  const edm::EDGetTokenT<PortableHostCollection<TrackTorchClassifierFeaturesSoA>> features_token_;
  const float min_score_;
  const float dxy_threshold_;
  const float high_dxy_min_score_;

  const edm::EDPutTokenT<reco::TrackCollection> filtered_tracks_token_;
  const edm::EDPutTokenT<std::vector<float>> scores_output_token_;
};

TrackTorchClassifierFromSoA::TrackTorchClassifierFromSoA(const edm::ParameterSet& iConfig)
    : tracks_token_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      scores_token_(consumes(iConfig.getParameter<edm::InputTag>("scores"))),
      features_token_(consumes(iConfig.getParameter<edm::InputTag>("features"))),
      min_score_(iConfig.getParameter<double>("minScore")),
      dxy_threshold_(iConfig.getParameter<double>("dxyThreshold")),
      high_dxy_min_score_(iConfig.getParameter<double>("highDxyMinScore")),
      filtered_tracks_token_(produces<reco::TrackCollection>()),
      scores_output_token_(produces<std::vector<float>>("MVAScores")) {}

void TrackTorchClassifierFromSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltInitialStepTracks"));
  desc.add<edm::InputTag>("scores", edm::InputTag("hltInitialStepTrackTorchClassifier"));
  desc.add<edm::InputTag>("features", edm::InputTag("hltInitialStepTrackTorchClassifier"));
  desc.add<double>("minScore", 0.5)->setComment("Minimum DNN score to keep track (working point)");
  desc.add<double>("dxyThreshold", 0.5)->setComment("Tracks with |dxy| > this value bypass the score cut");
  desc.add<double>("highDxyMinScore", 0.5)->setComment("Minimum DNN score to keep high dxy track (working point)");
  descriptions.addWithDefaultLabel(desc);
}

void TrackTorchClassifierFromSoA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& tracks = iEvent.get(tracks_token_);
  const auto& scores_host = iEvent.get(scores_token_);
  const auto& features_host = iEvent.get(features_token_);

  const auto nTracks = tracks.size();

  // Create filtered track collection
  auto filtered_tracks = std::make_unique<reco::TrackCollection>();
  auto all_scores = std::make_unique<std::vector<float>>();
  all_scores->reserve(nTracks);

  // Access scores and features from the host collection
  auto scores_view = scores_host.const_view();
  auto features_view = features_host.const_view();

  for (size_t i = 0; i < nTracks; ++i) {
    float score = scores_view[i].score();
    float dxy = features_view[i].dxyBeamSpot();
    all_scores->push_back(score);

    if (score >= min_score_ || ((std::abs(dxy) > dxy_threshold_) && (score >= high_dxy_min_score_))) {
      filtered_tracks->push_back(tracks[i]);
    }
  }

  iEvent.put(filtered_tracks_token_, std::move(filtered_tracks));
  iEvent.put(scores_output_token_, std::move(all_scores));
}

DEFINE_FWK_MODULE(TrackTorchClassifierFromSoA);
