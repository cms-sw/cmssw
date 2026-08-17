/**
 * PixelTrackTorchHighPuritySelector
 *
 * GPU/Accelerator module performing HighPurity pixel-track selection composed of:
 *   1. CA-based quality preselection
 *   2. Feature extraction
 *   3. TorchScript DNN inference
 *   4. Score-based filtering
 *   5. Track/hit compaction and output production
 *
 * Input: TracksSoA (pixel tracks + hit associations), TrackingRecHitsSoA + OTRecHitsSoA
 * (the latter only when useHitFeatures = true).
 *
 * Pipeline: TracksSoA -> CA preselection -> feature extraction -> Torch inference ->
 * score filtering -> output TrackSoA compaction.
 *
 * Torch inference: track tensor [maxPreselectedTracks, N_track_features], padding slots 0-filled.
 */

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"

#include <deque>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackTorchHighPuritySelectorKernels.h"

#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

// #define PIXEL_TRACK_HP_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /// Input/output tensors associated to a single inference batch.
  struct BatchIO {
    cms::torch::alpakatools::TensorCollection<Queue> inputs;
    cms::torch::alpakatools::TensorCollection<Queue> outputs;
  };

  class PixelTrackTorchHighPuritySelector : public stream::FixedQueueEDProducer<> {
    using TkSoADevice = reco::TracksSoACollection;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using OTHitsOnDevice = reco::OTRecHitsSoACollection;
    using TrackHitSoA = ::reco::TrackHitSoA;

  public:
    explicit PixelTrackTorchHighPuritySelector(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, const device::EventSetup&) override;
    void beginStream(edm::StreamID /*sid*/, Queue queue) override;

    /// Registers the "track_features" input block of `tc` for batch `i_batch`: the contiguous
    /// prefix run of PixelTrackFeaturesSoA the model consumes (17 fit/cov columns, plus the 14
    /// hit/stub columns when useHitFeatures). Order must match the TorchScript model input schema.
    /// Used by both warm-up and per-event passes.
    template <typename TRecord>
    static void addTrackFeatures(cms::torch::alpakatools::TensorCollection<Queue>& tc,
                                 int i_batch,
                                 TRecord& r,
                                 bool useHitFeatures) {
      if (useHitFeatures) {
        tc.add<PixelTrackFeaturesSoA>("track_features",
                                      i_batch,
                                      r.chi2(),
                                      r.dzError(),
                                      r.dxyError(),
                                      r.eta(),
                                      r.nHits(),
                                      r.phi(),
                                      r.phiError(),
                                      r.pt(),
                                      r.qOverPtError(),
                                      r.dzBS(),
                                      r.dxyBS(),
                                      r.nLayers(),
                                      r.cotThetaError(),
                                      r.covCotThetaDz(),
                                      r.covDxyQOverPt(),
                                      r.covPhiDxy(),
                                      r.covPhiQOverPt(),
                                      r.caFitChi2(),
                                      r.psFrac(),
                                      r.r0(),
                                      r.nPS(),
                                      r.spanZ(),
                                      r.nStubs(),
                                      r.logChi2Stub(),
                                      r.kErr(),
                                      r.dcaEst(),
                                      r.nBarrel(),
                                      r.rzChi2(),
                                      r.meanStubKappa(),
                                      r.leverArm(),
                                      r.rMax());
      } else {
        tc.add<PixelTrackFeaturesSoA>("track_features",
                                      i_batch,
                                      r.chi2(),
                                      r.dzError(),
                                      r.dxyError(),
                                      r.eta(),
                                      r.nHits(),
                                      r.phi(),
                                      r.phiError(),
                                      r.pt(),
                                      r.qOverPtError(),
                                      r.dzBS(),
                                      r.dxyBS(),
                                      r.nLayers(),
                                      r.cotThetaError(),
                                      r.covCotThetaDz(),
                                      r.covDxyQOverPt(),
                                      r.covPhiDxy(),
                                      r.covPhiQOverPt());
      }
    }

    /// Inference dtype, shared by the warm-up and the per-event forward passes: the FIRST forward
    /// is what moves (and casts) the model to the device, so the two must agree.
    std::optional<::torch::Dtype> inferenceDtype() const {
      return inferenceHalf_ ? std::optional<::torch::Dtype>{::torch::kHalf} : std::nullopt;
    }

    const device::EDGetToken<TkSoADevice> pixelTrackToken_;
    const int maxNumberOfTracks_;
    const int maxPreselectedTracks_;
    const int minNumberOfHits_;
    const int avgHitsPerTrack_;
    const pixelTrack::Quality minimumTrackQuality_;
    const double scoreThreshold_;
    // dxy-aware threshold ramp: threshold goes from scoreThresholdLowDxy at |dxyBS|=0 down to
    // scoreThreshold at |dxyBS|>=dxyRampKnee. scoreThresholdLowDxy < 0 disables it (flat threshold).
    const double scoreThresholdLowDxy_;
    const double dxyRampKnee_;
    // Hit/stub features: when true, the 14 CA hit/stub features are appended to the 17 fit/cov
    // features -> the enriched 31-feature model. False selects the plain 17-feature path. The
    // merged-hits product is consumed ONLY when enabled.
    const bool useHitFeatures_;
    // forward(dtype) casts the WHOLE model. kHalf is fine for the MLP but BREAKS a tree model
    // (casts its int64 index buffers to half). inferenceHalf=false -> no cast -> native fp32.
    const bool inferenceHalf_;
    device::EDGetToken<HitsOnDevice> mergedHitsToken_;
    // Raw OT-rechit SoA, so the hit-feature walk can resolve the bit30-tagged raw-OT ids a track
    // may carry. Consumed only when useHitFeatures_ (same guard as mergedHitsToken_).
    device::EDGetToken<OTHitsOnDevice> otRecHitsSoAToken_;
    torch::AlpakaModel model_;
    const int batchSize_;
    const int warmupIterations_ = 3;
    const device::EDPutToken<TkSoADevice> tokenTrackOut_;
  };

  PixelTrackTorchHighPuritySelector::PixelTrackTorchHighPuritySelector(const edm::ParameterSet& iConfig)
      : FixedQueueEDProducer(iConfig),
        pixelTrackToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))),
        maxNumberOfTracks_(iConfig.getParameter<int>("maxNumberOfTracks")),
        maxPreselectedTracks_(iConfig.getParameter<int>("maxPreselectedTracks")),
        minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
        avgHitsPerTrack_(iConfig.getParameter<int>("avgHitsPerTrack")),
        minimumTrackQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minimumTrackQuality"))),
        scoreThreshold_(iConfig.getParameter<double>("scoreThreshold")),
        scoreThresholdLowDxy_(iConfig.getParameter<double>("scoreThresholdLowDxy")),
        dxyRampKnee_(iConfig.getParameter<double>("dxyRampKnee")),
        useHitFeatures_(iConfig.getParameter<bool>("useHitFeatures")),
        inferenceHalf_(iConfig.getParameter<bool>("inferenceHalf")),
        model_(iConfig.getParameter<edm::FileInPath>("model").fullPath()),
        batchSize_(iConfig.getParameter<int>("batchSize")),
        tokenTrackOut_(produces()) {
    if (useHitFeatures_) {
      mergedHitsToken_ = consumes(iConfig.getParameter<edm::InputTag>("mergedHitsSrc"));
      otRecHitsSoAToken_ = consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSoASrc"));
    }
    if (minimumTrackQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minimumTrackQuality") + " is not a pixelTrack::Quality";
    }
    if (minimumTrackQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minimumTrackQuality") + " not supported";
    }
    if (maxPreselectedTracks_ > maxNumberOfTracks_) {
      throw cms::Exception("PixelTrackConfiguration") << "maxPreselectedTracks must be <= maxNumberOfTracks";
    }
  }

  void PixelTrackTorchHighPuritySelector::beginStream(edm::StreamID /*sid*/, Queue queue) {
    // Warmup the model with dummy data

    // Create temporary feature and score buffers used to warm up the model.
    PixelTrackFeaturesOnDevice trackFeatures(queue, batchSize_);
    PixelTrackScoresOnDevice trackScoresOnDevice(queue, batchSize_);
    auto track_record = trackFeatures.view().records();
    auto score_record = trackScoresOnDevice.view().records();

    for (auto it = 0; it < warmupIterations_; ++it) {
      cms::torch::alpakatools::TensorCollection<Queue> dummy_inputs(batchSize_);
      cms::torch::alpakatools::TensorCollection<Queue> dummy_outputs(batchSize_);

      // Same column list (and therefore the same tensor width) as the per-event inference below.
      addTrackFeatures(dummy_inputs, 0, track_record, useHitFeatures_);

      dummy_outputs.add<PixelTrackScoresSoA>("track_scores", score_record.score());

      model_.forward(queue, dummy_inputs, dummy_outputs, inferenceDtype());
    }
  }

  void PixelTrackTorchHighPuritySelector::produce(device::Event& iEvent, const device::EventSetup&) {
    /*
    Processing steps:
      1. CA-based preselection of tracks
      2. Feature extraction (track SoA)
      3. DNN inference
      4. Score-based filtering
      5. Track compaction and output production
*/
    // Retrieve tokens
    auto& queue = iEvent.queue();
    const auto& tracks = iEvent.get(pixelTrackToken_).view();

    // Instantiate the necessary objects in memory
    //  - Temporary storage for filtering
    auto d_nPreselectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_nSelectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_preselectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);
    auto d_selectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_trackHitCounts = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_selectedTrackHitOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_preselectionOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);

    alpaka::memset(queue, d_nPreselectedTracks, 0);
    alpaka::memset(queue, d_nSelectedTracks, 0);
    alpaka::memset(queue, d_trackHitCounts, 0);
    alpaka::memset(queue, d_selectedTrackHitOffsets, 0);
    alpaka::memset(queue, d_preselectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_selectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_preselectionOffsets, 0);

    //  - Features and scores containers
    PixelTrackFeaturesOnDevice trackFeatures(queue, maxPreselectedTracks_);
    PixelTrackScoresOnDevice trackScoresOnDevice(queue, maxPreselectedTracks_);

    // Optional debug definitions
#ifdef PIXEL_TRACK_HP_DEBUG
    auto h_nPreselectedTracks = cms::alpakatools::make_host_buffer<int>(queue);
    auto h_nSelectedTracks = cms::alpakatools::make_host_buffer<int>(queue);
    auto nPreselectedTracks = 0;
    auto nSelectedTracks = 0;
    // Helper to copy the number of kept tracks back to host (debug only)
    auto fetchNumPreselectedTracks = [&]() {
      alpaka::memcpy(queue, h_nPreselectedTracks, d_nPreselectedTracks);
      alpaka::wait(queue);
      return *h_nPreselectedTracks;
    };
    auto fetchNumSelectedTracks = [&]() {
      alpaka::memcpy(queue, h_nSelectedTracks, d_nSelectedTracks);
      alpaka::wait(queue);
      return *h_nSelectedTracks;
    };
#endif

    // 1. CA-based preselection of tracks
    //  Launch first kernel to look which tracks need to be filtered out
    //  based on quality criteria from the CA

    launchCAPreselection(queue,
                         maxNumberOfTracks_,
                         minNumberOfHits_,
                         minimumTrackQuality_,
                         tracks.tracks(),
                         alpaka::getPtrNative(d_preselectedTrackIndices),
                         alpaka::getPtrNative(d_preselectionOffsets),
                         alpaka::getPtrNative(d_nPreselectedTracks));

#ifdef PIXEL_TRACK_HP_DEBUG
    nPreselectedTracks = fetchNumPreselectedTracks();
    std::cout << "PixelTrackTorchHighPuritySelector::Prefiltered tracks=" << nPreselectedTracks << "\n";
#endif

    // 2. Feature extraction. The merged TrackingRecHitsSoA is needed ONLY for the hit/stub
    // features (when enabled); otherwise pass a null view + nHitsTot=0.
    ::reco::TrackingRecHitConstView mergedHitsView{};
    int nHitsTot = 0;
    // OT-rechit view for resolving tagged OT extras (empty view + 0 when not needed).
    ::reco::OTRecHitsConstView otHitsView{};
    uint32_t nOTHits = 0;
    if (useHitFeatures_) {
      const auto& mergedHits = iEvent.get(mergedHitsToken_);
      mergedHitsView = mergedHits.view().trackingHits();
      nHitsTot = mergedHitsView.metadata().size();
      const auto& otHits = iEvent.get(otRecHitsSoAToken_);
      otHitsView = otHits.const_view().otRecHits();
      nOTHits = otHitsView.metadata().size();
    }

    launchFeaturesExtractor(queue,
                            maxPreselectedTracks_,
                            tracks.tracks(),
                            tracks.trackHits(),
                            mergedHitsView,
                            nHitsTot,
                            otHitsView,
                            nOTHits,
                            useHitFeatures_,
                            alpaka::getPtrNative(d_preselectedTrackIndices),
                            alpaka::getPtrNative(d_nPreselectedTracks),
                            trackFeatures.view(),
                            alpaka::getPtrNative(d_trackHitCounts));

    // 3. DNN inference
    //  Prepare TensorCollection inputs and outputs for the model
    auto track_record = trackFeatures.view().records();
    auto score_record = trackScoresOnDevice.view().records();
    const auto n_batches = (maxPreselectedTracks_ + batchSize_ - 1) / batchSize_;
    std::deque<BatchIO> batches;

    // - Tensor collections for DNN inference
    for (auto i_batch = 0; i_batch < n_batches; ++i_batch) {
      batches.emplace_back(
          BatchIO{cms::torch::alpakatools::TensorCollection<Queue>(batchSize_, maxPreselectedTracks_),
                  cms::torch::alpakatools::TensorCollection<Queue>(batchSize_, maxPreselectedTracks_)});

      auto& batch = batches.back();
      // Order must match the TorchScript model input schema
      addTrackFeatures(batch.inputs, i_batch, track_record, useHitFeatures_);

      batch.outputs.add<PixelTrackScoresSoA>("track_scores", i_batch, score_record.score());

      model_.forward(queue, batch.inputs, batch.outputs, inferenceDtype());
    }

    // 4. Score-based filtering
    launchScoreFilter(queue,
                      maxPreselectedTracks_,
                      scoreThreshold_,
                      scoreThresholdLowDxy_,
                      dxyRampKnee_,
                      trackFeatures.const_view(),
                      trackScoresOnDevice.view(),
                      alpaka::getPtrNative(d_preselectedTrackIndices),
                      alpaka::getPtrNative(d_nPreselectedTracks),
                      alpaka::getPtrNative(d_trackHitCounts),
                      alpaka::getPtrNative(d_selectedTrackIndices),
                      alpaka::getPtrNative(d_nSelectedTracks),
                      alpaka::getPtrNative(d_selectedTrackHitOffsets));

#ifdef PIXEL_TRACK_HP_DEBUG
    nSelectedTracks = fetchNumSelectedTracks();
    std::cout << "PixelTrackTorchHighPuritySelector::Filtered tracks=" << nSelectedTracks << "\n";
#endif

    auto tracks_out = launchProduceOutputTracks(queue,
                                                maxPreselectedTracks_,
                                                avgHitsPerTrack_,
                                                tracks.tracks(),
                                                tracks.trackHits(),
                                                alpaka::getPtrNative(d_selectedTrackIndices),
                                                alpaka::getPtrNative(d_nSelectedTracks),
                                                alpaka::getPtrNative(d_selectedTrackHitOffsets));
    iEvent.emplace(tokenTrackOut_, std::move(tracks_out));
  }

  void PixelTrackTorchHighPuritySelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pixelTrackSrc", {"hltPhase2PixelTracksSoA"});
    desc.add<int>("maxNumberOfTracks", 100000);
    desc.add<int>("maxPreselectedTracks", 10000);
    desc.add<int>("minNumberOfHits", 0);
    desc.add<int>("avgHitsPerTrack", 8);
    desc.add<std::string>("minimumTrackQuality", "tight");
    desc.add<edm::FileInPath>("model");
    desc.add<double>("scoreThreshold", 0.5);
    desc.add<int>("batchSize", 10);
    // dxy-aware threshold ramp: scoreThresholdLowDxy<0 disables it (flat scoreThreshold). When
    // >=0, the cut ramps from scoreThresholdLowDxy at |dxyBS|=0 to scoreThreshold at
    // |dxyBS|>=dxyRampKnee[cm] -- a more aggressive cut at low reco displacement.
    desc.add<double>("scoreThresholdLowDxy", -1.0);
    desc.add<double>("dxyRampKnee", 2.0);
    // Hit/stub features (enriched model). Default false = 17-feature path.
    // mergedHitsSrc is the merged TrackingRecHitsSoA the CA indexed -- only consumed when useHitFeatures=true.
    desc.add<bool>("useHitFeatures", false);
    desc.add<edm::InputTag>("mergedHitsSrc", {"hltPhase2PixelRecHitsStubsMerger"});
    // Raw OT-rechit SoA (only consumed when useHitFeatures=true): resolves the bit30-tagged
    // raw-OT hit ids so OT-extended tracks are scored on their full hit content.
    desc.add<edm::InputTag>("otRecHitsSoASrc", {"hltPixelSeedingOTRecHitsSoA"});
    // inferenceHalf=false keeps the model in fp32 (required for a tree/Hummingbird model whose
    // int index buffers must not be cast to half); true = the fp16 path used by the MLP models.
    desc.add<bool>("inferenceHalf", true);
    descriptions.addWithDefaultLabel(desc);
  }
};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTrackTorchHighPuritySelector);
