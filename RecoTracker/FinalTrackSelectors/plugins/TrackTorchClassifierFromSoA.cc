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
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// This module consumes the HOST copy of the Alpaka device scores
// The framework automatically creates host copies of device PortableCollections
class TrackTorchClassifierFromSoA : public edm::stream::EDProducer<> {
public:
  explicit TrackTorchClassifierFromSoA(const edm::ParameterSet& iConfig);
  ~TrackTorchClassifierFromSoA() override = default;

  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const edm::EDGetTokenT<reco::TrackCollection> tracks_token_;
  const edm::EDGetTokenT<PortableHostCollection<TrackTorchClassifierScoresSoA>> scores_token_;
  const edm::EDGetTokenT<PortableHostCollection<TrackTorchClassifierFeaturesSoA>> features_token_;
  // qualitycuts (loose, tight, hp) for prompt/displaced
  const std::vector<double> quality_cuts_prompt_;
  const float dxy_threshold_;
  const std::vector<double> quality_cuts_displaced_;

  const edm::EDPutTokenT<MVACollection> scores_output_token_;
  const edm::EDPutTokenT<edm::ValueMap<float>> mva_vals_token_;
  const edm::EDPutTokenT<edm::ValueMap<int>> track_quals_token_;
  const edm::EDPutTokenT<QualityMaskCollection> quality_mask_output_token_;
};

TrackTorchClassifierFromSoA::TrackTorchClassifierFromSoA(const edm::ParameterSet& iConfig)
    : tracks_token_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      scores_token_(consumes(iConfig.getParameter<edm::InputTag>("scores"))),
      features_token_(consumes(iConfig.getParameter<edm::InputTag>("features"))),
      quality_cuts_prompt_(iConfig.getParameter<std::vector<double>>("qualityCutsPrompt")),
      dxy_threshold_(iConfig.getParameter<double>("dxyThreshold")),
      quality_cuts_displaced_(iConfig.getParameter<std::vector<double>>("qualityCutsDisplaced")),
      scores_output_token_(produces<MVACollection>("MVAValues")),
      mva_vals_token_(produces<edm::ValueMap<float>>("MVAVals")),
      track_quals_token_(produces<edm::ValueMap<int>>("TrackQuals")),
      quality_mask_output_token_(produces<QualityMaskCollection>("QualityMasks")) {
  assert(quality_cuts_prompt_.size() == 3);
  assert(quality_cuts_displaced_.size() == 3);
}

void TrackTorchClassifierFromSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltInitialStepTracks"));
  desc.add<edm::InputTag>("scores", edm::InputTag("hltInitialStepTrackTorchClassifier"));
  desc.add<edm::InputTag>("features", edm::InputTag("hltInitialStepTrackTorchClassifier"));
  desc.add<std::vector<double>>("qualityCutsPrompt", {0.5, 0.5, 0.5})
      ->setComment("MVA score quality cuts for prompt tracks");
  desc.add<double>("dxyThreshold", 0.5)->setComment("Tracks with |dxy| > this value use the displaced quality cuts");
  desc.add<std::vector<double>>("qualityCutsDisplaced", {0.5, 0.5, 0.5})
      ->setComment("MVA score quality cuts for displaced tracks");
  descriptions.addWithDefaultLabel(desc);
}

void TrackTorchClassifierFromSoA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::TrackCollection> tracks_handle;
  iEvent.getByToken(tracks_token_, tracks_handle);
  const auto& tracks = *tracks_handle;
  const auto& scores_host = iEvent.get(scores_token_);
  const auto& features_host = iEvent.get(features_token_);
  const auto nTracks = tracks.size();

  // Create products: MVA score collection and quality mask collection
  auto mvas = std::make_unique<MVACollection>(tracks.size(), -99.f);
  auto mva_vals = std::make_unique<edm::ValueMap<float>>();
  std::vector<int> selTracks(nTracks, 0);
  auto selTracksValueMap = std::make_unique<edm::ValueMap<int>>();

  // Access scores and features from the host collection
  auto scores_view = scores_host.const_view();
  auto features_view = features_host.const_view();

  // Loop over tracks
  for (size_t i = 0; i < nTracks; ++i) {
    float score = scores_view[i].score();
    float dxy = features_view[i].dxyBeamSpot();
    int quality = tracks[i].qualityMask();

    (*mvas)[i] = score;
    const bool pass_loose =
        score >= quality_cuts_prompt_[0] || ((std::abs(dxy) > dxy_threshold_) && (score >= quality_cuts_displaced_[0]));
    const bool pass_tight =
        score >= quality_cuts_prompt_[1] || ((std::abs(dxy) > dxy_threshold_) && (score >= quality_cuts_displaced_[1]));
    const bool pass_hp =
        score >= quality_cuts_prompt_[2] || ((std::abs(dxy) > dxy_threshold_) && (score >= quality_cuts_displaced_[2]));

    if (pass_loose)
      quality |= (1 << reco::TrackBase::loose);
    if (pass_loose && pass_tight)
      quality |= (1 << reco::TrackBase::tight);
    if (pass_loose && pass_tight && pass_hp)
      quality |= (1 << reco::TrackBase::highPurity);
    selTracks[i] = quality;
  }

  edm::ValueMap<float>::Filler mva_filler(*mva_vals);
  mva_filler.insert(tracks_handle, mvas->begin(), mvas->end());
  mva_filler.fill();

  edm::ValueMap<int>::Filler qual_filler(*selTracksValueMap);
  qual_filler.insert(tracks_handle, selTracks.begin(), selTracks.end());
  qual_filler.fill();

  for (auto& q : selTracks)
    q = std::max(q, 0);
  auto quals = std::make_unique<QualityMaskCollection>(selTracks.begin(), selTracks.end());
  iEvent.put(scores_output_token_, std::move(mvas));
  iEvent.put(mva_vals_token_, std::move(mva_vals));
  iEvent.put(quality_mask_output_token_, std::move(quals));
  iEvent.put(track_quals_token_, std::move(selTracksValueMap));
}

DEFINE_FWK_MODULE(TrackTorchClassifierFromSoA);
