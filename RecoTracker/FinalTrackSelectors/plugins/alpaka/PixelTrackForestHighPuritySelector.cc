/**
 * PixelTrackForestHighPuritySelector
 *
 * HighPurity pixel-track selection with a gradient-boosted decision tree: CA-based quality
 * preselection, feature extraction, compact forest inference (alpaka traversal kernel), score cut,
 * track/hit compaction. The XGBoost forest is exported to a compact binary (int8 split feature,
 * fp32 threshold/leaf, int32 children, per-tree root offsets, base logit), loaded once per process
 * into a GlobalCache (DispTreeCache) and lifted to the device once.
 */

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include <cstdint>
#include <fstream>
#include <optional>
#include <utility>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaCore/interface/CopyToDeviceCache.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

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
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "RecoTracker/PixelSeeding/interface/CAHitsView.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackForestHighPuritySelectorKernels.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackTorchHighPuritySelectorKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Per-process shared compact gradient-boosted tree, loaded into host buffers and lifted by
  // CopyToDeviceCache to every visible device; get(queue) returns the buffer resident on that
  // queue's device. Read-only, so concurrency-safe with no mutex. Per-device is required: the
  // framework spreads streams across all visible GPUs.
  // Binary format: int32 nNodes, int32 nTrees, float baseLogit, then int8 feat[N] (-1 = leaf),
  // float val[N] (threshold / leaf value), int32 left[N], int32 right[N], int32 roots[nTrees].
  template <typename T>
  using TreeArrayCache = cms::alpakatools::CopyToDeviceCache<Device, cms::alpakatools::host_buffer<T[]>>;

  struct DispTreeCache {
    explicit DispTreeCache(std::string const& path) : DispTreeCache(load(path)) {}

    int nNodes = 0, nTrees = 0;
    float baseLogit = 0.f;
    TreeArrayCache<int8_t> feat;
    TreeArrayCache<float> val;
    TreeArrayCache<int32_t> left, right, roots;

  private:
    // Host-side staging read from the binary; the public constructor delegates to the private one
    // below, which uploads it.
    struct HostTree {
      int nNodes, nTrees;
      float baseLogit;
      cms::alpakatools::host_buffer<int8_t[]> feat;
      cms::alpakatools::host_buffer<float[]> val;
      cms::alpakatools::host_buffer<int32_t[]> left, right, roots;
    };
    static HostTree load(std::string const& path) {
      std::ifstream in(path, std::ios::binary);
      if (!in)
        throw cms::Exception("PixelTrackConfiguration") << "cannot open compact tree binary: " << path;
      int32_t nN = 0, nt = 0;
      float bl = 0.f;
      in.read(reinterpret_cast<char*>(&nN), 4);
      in.read(reinterpret_cast<char*>(&nt), 4);
      in.read(reinterpret_cast<char*>(&bl), 4);
      // Header sanity check, before the allocations it sizes: a negative count from the file
      // becomes a huge size_t in make_host_buffer.
      if (!in || nN <= 0 || nt <= 0)
        throw cms::Exception("PixelTrackConfiguration")
            << "compact tree binary " << path << ": bad header (nNodes=" << nN << ", nTrees=" << nt << ").";
      auto feat = cms::alpakatools::make_host_buffer<int8_t[]>(nN);
      auto val = cms::alpakatools::make_host_buffer<float[]>(nN);
      auto left = cms::alpakatools::make_host_buffer<int32_t[]>(nN);
      auto right = cms::alpakatools::make_host_buffer<int32_t[]>(nN);
      auto roots = cms::alpakatools::make_host_buffer<int32_t[]>(nt);
      in.read(reinterpret_cast<char*>(feat.data()), std::streamsize(nN) * sizeof(int8_t));
      in.read(reinterpret_cast<char*>(val.data()), std::streamsize(nN) * sizeof(float));
      in.read(reinterpret_cast<char*>(left.data()), std::streamsize(nN) * sizeof(int32_t));
      in.read(reinterpret_cast<char*>(right.data()), std::streamsize(nN) * sizeof(int32_t));
      in.read(reinterpret_cast<char*>(roots.data()), std::streamsize(nt) * sizeof(int32_t));
      if (!in)
        throw cms::Exception("PixelTrackConfiguration") << "compact tree binary truncated/corrupt: " << path;
      // The scorer reads f[0..kNPixelTrackFeatures-1] with no bound check of its own, so an index
      // >= width is an out-of-bounds device read and a value below -1 is mistaken for a leaf
      // (traversal tests `>= 0`). Models narrower than the current width stay valid.
      for (int n = 0; n < nN; ++n) {
        const int fi = feat[n];
        if (fi >= kNPixelTrackFeatures || fi < -1)
          throw cms::Exception("PixelTrackConfiguration")
              << "compact tree binary " << path << ": node " << n << " splits on feature index " << fi
              << ", outside the [0, " << kNPixelTrackFeatures - 1 << "] range the selector provides (-1 = leaf). "
              << "The model was exported against a different feature ABI than PixelTrackFeaturesSoA's "
              << kNPixelTrackFeatures << " columns.";
      }
      // The traversal follows treeRoots/treeLeft/treeRight with no bound check of its own, so a
      // corrupt index is an out-of-bounds device read. A negative left/right is a valid leaf marker;
      // a root must be a real node.
      for (int t = 0; t < nt; ++t)
        if (roots[t] < 0 || roots[t] >= nN)
          throw cms::Exception("PixelTrackConfiguration")
              << "compact tree binary " << path << ": tree " << t << " has root node " << roots[t] << ", outside [0, "
              << nN << ").";
      for (int n = 0; n < nN; ++n)
        if (left[n] >= nN || right[n] >= nN)
          throw cms::Exception("PixelTrackConfiguration")
              << "compact tree binary " << path << ": node " << n << " has children (" << left[n] << ", " << right[n]
              << "), outside [0, " << nN << ").";
      return HostTree{nN, nt, bl, std::move(feat), std::move(val), std::move(left), std::move(right), std::move(roots)};
    }
    explicit DispTreeCache(HostTree h)
        : nNodes(h.nNodes),
          nTrees(h.nTrees),
          baseLogit(h.baseLogit),
          feat(h.feat),
          val(h.val),
          left(h.left),
          right(h.right),
          roots(h.roots) {}
  };

  // SynchronizingEDProducer (two-phase acquire/produce) rather than FixedQueueEDProducer, which
  // refuses ExternalWork. All of one event's work still runs on one queue: the queue created in
  // acquire is carried into produce.
  class PixelTrackForestHighPuritySelector : public stream::SynchronizingEDProducer<edm::GlobalCache<DispTreeCache>> {
    using TkSoADevice = reco::TracksSoACollection;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using StubsOnDevice = reco::StubsSoACollection;
    using OTHitsOnDevice = reco::OTRecHitsSoACollection;

  public:
    explicit PixelTrackForestHighPuritySelector(const edm::ParameterSet&, const DispTreeCache*);
    static std::unique_ptr<DispTreeCache> initializeGlobalCache(const edm::ParameterSet& iConfig) {
      return std::make_unique<DispTreeCache>(iConfig.getParameter<edm::FileInPath>("model").fullPath());
    }
    static void globalEndJob(DispTreeCache*) {}
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // acquire() runs the whole selection chain asynchronously and issues one async D2H of the
    // two counts the compaction kernel commits. produce() reads the landed counts (plain host
    // memory by then -- no wait), emplaces the collection and publishes the counts as host products.
    void acquire(device::Event const&, device::EventSetup const&) override;
    void produce(device::Event&, device::EventSetup const&) override;

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
    // When true, the CA hit/stub, provenance and pixel-cluster columns are appended to the 17
    // fit/cov features. The constructor requires true; the merged-hits product is consumed only
    // then.
    const bool useHitFeatures_;
    device::EDGetToken<HitsOnDevice> pixelHitsToken_;
    device::EDGetToken<StubsOnDevice> stubsToken_;
    // Raw OT-rechit SoA, so the hit-feature walk can resolve the bit30-tagged raw-OT ids a track
    // may carry. Consumed only when useHitFeatures_.
    device::EDGetToken<OTHitsOnDevice> otRecHitsSoAToken_;
    const device::EDPutToken<TkSoADevice> tokenTrackOut_;
    // Counts actually selected, so a consumer can size against them instead of against this
    // collection's capacity. Both are exact by construction.
    const edm::EDPutTokenT<uint32_t> tokenNTracksOut_;
    const edm::EDPutTokenT<uint32_t> tokenNKeptHitsOut_;

    // State carried from acquire() to produce() (per stream): pendingTracks_ is built in acquire and
    // emplaced in produce; countsHost_ is a 2-word host mirror ([0] tracks, [1] hits) whose async copy
    // is issued at the end of acquire. produce() releases both.
    std::optional<TkSoADevice> pendingTracks_;
    std::optional<cms::alpakatools::host_buffer<uint32_t[]>> countsHost_;
  };

  PixelTrackForestHighPuritySelector::PixelTrackForestHighPuritySelector(const edm::ParameterSet& iConfig,
                                                                         const DispTreeCache*)
      : SynchronizingEDProducer(iConfig),
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
        tokenTrackOut_(produces()),
        tokenNTracksOut_(produces("nTracks")),
        tokenNKeptHitsOut_(produces("nKeptHits")) {
    if (useHitFeatures_) {
      pixelHitsToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
      stubsToken_ = consumes(iConfig.getParameter<edm::InputTag>("stubsSrc"));
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
    // The tree scorer reads all columns unconditionally; columns 18-42 are written only under
    // useHitFeatures, so refuse useHitFeatures=false (would score uninitialised memory).
    if (!useHitFeatures_) {
      throw cms::Exception("PixelTrackConfiguration")
          << "PixelTrackForestHighPuritySelector requires useHitFeatures = True (the tree scorer "
             "reads the full feature vector, whose columns 18 and above are hit-derived)";
    }
  }

  void PixelTrackForestHighPuritySelector::acquire(device::Event const& iEvent, device::EventSetup const&) {
    // Start from an empty carried state.
    pendingTracks_.reset();
    countsHost_.reset();

    auto& queue = iEvent.queue();
    const auto& tracks = iEvent.get(pixelTrackToken_).view();

    // Temporary storage for filtering
    auto d_nPreselectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_nSelectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_preselectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);
    auto d_selectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_trackHitCounts = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_selectedTrackHitOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_preselectionOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);

    // No pre-fill of the buffers above: each is written before read over a range containing every
    // index any consumer reads. The buffers whose consumer can read an unwritten element
    // (preselectionMask, selectionMask, selectedTrackHitCounts) are filled where they are
    // allocated.

    // Features and scores containers
    PixelTrackFeaturesOnDevice trackFeatures(queue, maxPreselectedTracks_);
    PixelTrackScoresOnDevice trackScoresOnDevice(queue, maxPreselectedTracks_);

    // 1. CA-based preselection of tracks
    launchCAPreselection(queue,
                         maxNumberOfTracks_,
                         minNumberOfHits_,
                         minimumTrackQuality_,
                         tracks.tracks(),
                         alpaka::getPtrNative(d_preselectedTrackIndices),
                         alpaka::getPtrNative(d_preselectionOffsets),
                         alpaka::getPtrNative(d_nPreselectedTracks));

    // 2. Feature extraction. The merged TrackingRecHitsSoA (the product the CA indexed: its
    // trackHits().id() point into it) is needed only for the hit/stub features; otherwise an empty
    // view and nHitsTot = 0 are passed.
    caStructures::CAHitsView hitsView{};
    int nHitsTot = 0;
    // OT-rechit view for resolving tagged OT extras (empty view + 0 when not needed).
    ::reco::OTRecHitsConstView otHitsView{};
    uint32_t nOTHits = 0;
    if (useHitFeatures_) {
      const auto& pixHits = iEvent.get(pixelHitsToken_);
      const auto& stubs = iEvent.get(stubsToken_);
      hitsView = caStructures::CAHitsView(pixHits.const_view().trackingHits(),
                                          pixHits.const_view().hitModules(),
                                          stubs.const_view().stubs(),
                                          stubs.const_view().stubModules(),
                                          pixHits.nHits(),
                                          stubs.nStubs(),
                                          pixHits.nModules());
      nHitsTot = hitsView.size();
      const auto& otHits = iEvent.get(otRecHitsSoAToken_);
      otHitsView = otHits.const_view().otRecHits();
      nOTHits = otHitsView.metadata().size();
    }
    launchFeaturesExtractor(queue,
                            maxPreselectedTracks_,
                            tracks.tracks(),
                            tracks.trackHits(),
                            hitsView,
                            nHitsTot,
                            otHitsView,
                            nOTHits,
                            useHitFeatures_,
                            alpaka::getPtrNative(d_preselectedTrackIndices),
                            alpaka::getPtrNative(d_nPreselectedTracks),
                            trackFeatures.view(),
                            alpaka::getPtrNative(d_trackHitCounts));

    // 3. Tree inference, reading the buffers resident on this queue's device from the per-device
    // shared cache.
    auto const& tc = *globalCache();
    launchTreeScore(queue,
                    maxPreselectedTracks_,
                    alpaka::getPtrNative(tc.feat.get(queue)),
                    alpaka::getPtrNative(tc.val.get(queue)),
                    alpaka::getPtrNative(tc.left.get(queue)),
                    alpaka::getPtrNative(tc.right.get(queue)),
                    alpaka::getPtrNative(tc.roots.get(queue)),
                    tc.nTrees,
                    tc.baseLogit,
                    trackFeatures.const_view(),
                    alpaka::getPtrNative(d_nPreselectedTracks),
                    trackScoresOnDevice.view());

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

    // 2-word device scratch for the counts the compaction kernel commits. No pre-fill:
    // PixelTrackFilterKernel assigns both words under once_per_grid before the copy. Function scope
    // is safe: the caching allocator re-hands a freed block only after the queue's free event.
    auto d_selectedCounts = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2u);
    auto tracks_out = launchProduceOutputTracks(queue,
                                                maxPreselectedTracks_,
                                                avgHitsPerTrack_,
                                                tracks.tracks(),
                                                tracks.trackHits(),
                                                alpaka::getPtrNative(d_selectedTrackIndices),
                                                alpaka::getPtrNative(d_nSelectedTracks),
                                                alpaka::getPtrNative(d_selectedTrackHitOffsets),
                                                alpaka::getPtrNative(d_selectedCounts));
    // Async 8-byte D2H into a pinned buffer; the framework's between-phase drain delivers it.
    countsHost_.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 2u));
    alpaka::memcpy(queue, *countsHost_, d_selectedCounts);
    pendingTracks_.emplace(std::move(tracks_out));
  }

  void PixelTrackForestHighPuritySelector::produce(device::Event& iEvent, device::EventSetup const&) {
    // The counts landed while the framework waited on this event's queue: plain host memory here.
    uint32_t nTracksSel = 0;
    uint32_t nKeptHitsSel = 0;
    if (countsHost_) {
      nTracksSel = countsHost_->data()[0];
      nKeptHitsSel = countsHost_->data()[1];
    }

    if (pendingTracks_) {
      iEvent.emplace(tokenTrackOut_, std::move(*pendingTracks_));
    } else {
      // Unreachable while acquire() has no early-out; guarantees all three products always exist.
      auto& queue = iEvent.queue();
      TkSoADevice empty(queue, 0, 0);
      auto nTracks_d = cms::alpakatools::make_device_view(queue, empty.view().tracks().nTracks());
      alpaka::memset(queue, nTracks_d, 0);
      iEvent.emplace(tokenTrackOut_, std::move(empty));
    }
    iEvent.emplace(tokenNTracksOut_, nTracksSel);
    iEvent.emplace(tokenNKeptHitsOut_, nKeptHitsSel);

    pendingTracks_.reset();
    countsHost_.reset();
  }

  void PixelTrackForestHighPuritySelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pixelTrackSrc", {"hltPhase2PixelTracksSoA"});
    desc.add<int>("maxNumberOfTracks", 100000);
    desc.add<int>("maxPreselectedTracks", 10000);
    desc.add<int>("minNumberOfHits", 0);
    desc.add<int>("avgHitsPerTrack", 8);
    desc.add<std::string>("minimumTrackQuality", "tight");
    // Compact gradient-boosted tree binary, not a TorchScript .pt. Split-feature indices are
    // positions in the PixelTrackFeaturesSoA column order.
    desc.add<edm::FileInPath>("model");
    desc.add<double>("scoreThreshold", 0.5);
    // dxy-aware threshold ramp: scoreThresholdLowDxy < 0 disables it (flat scoreThreshold). When
    // >= 0 the cut ramps from scoreThresholdLowDxy at |dxyBS| = 0 to scoreThreshold at
    // |dxyBS| >= dxyRampKnee (cm), so low reco displacement is cut harder.
    desc.add<double>("scoreThresholdLowDxy", -1.0);
    desc.add<double>("dxyRampKnee", 2.0);
    // This module requires useHitFeatures = True. the hit features read the pixel rechits + stubs
    // the CA indexed, consumed only then.
    desc.add<bool>("useHitFeatures", true);
    desc.add<edm::InputTag>("pixelRecHitSrc", {"hltPhase2SiPixelRecHitsSoA"});
    desc.add<edm::InputTag>("stubsSrc", {"hltOTStubProducer"});
    // Raw OT-rechit SoA: resolves the bit30-tagged raw-OT hit ids so OT-extended tracks are scored
    // on their full hit content.
    desc.add<edm::InputTag>("otRecHitsSoASrc", {"hltPixelSeedingOTRecHitsSoA"});
    descriptions.addWithDefaultLabel(desc);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTrackForestHighPuritySelector);
