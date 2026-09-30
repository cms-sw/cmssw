#include <alpaka/alpaka.hpp>

#include <numeric>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "TrackSoAMergerKernels.h"

// #define GPU_DEBUG
// #define NTRACKS_DEBUG
// #define DUPLICATE_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TracksSoAMerger : public stream::SynchronizingEDProducer<> {
    using Algo = TrackSoAMergerKernels;
    using AlgoParams = ::mergerKernels::Params;
    using TracksConstView = ::mergerKernels::TracksConstView;
    using TrackHitsConstView = ::mergerKernels::TrackHitsConstView;
    using TracksMultiView = ::mergerKernels::TracksMultiView;
    using TrackHitsMultiView = ::mergerKernels::TrackHitsMultiView;
    using Tracks = reco::TracksSoACollection;

  public:
    explicit TracksSoAMerger(const edm::ParameterSet& iConfig);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    // merging parameters
    pixelTrack::Quality const minQuality_;
    double const matchFraction_;
    int const dupMinHits_;
    float const dupNSigma2_;
    float const dupMaxDeltaR2_;
    float const dupPtDifference_;

    // tokens
    std::vector<device::EDGetToken<Tracks>> trackTokens_;
    std::vector<edm::InputTag> trackTags_;
    const device::EDPutToken<Tracks> outputTracks_;

    // output tracks
    std::optional<Tracks> tracks_d_;
  };

  TracksSoAMerger::TracksSoAMerger(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        matchFraction_(iConfig.getParameter<double>("matchFraction")),
        dupMinHits_(iConfig.getParameter<int>("minHitsForDuplicate")),
        dupNSigma2_(iConfig.getParameter<double>("dupNSigma2")),
        dupMaxDeltaR2_(iConfig.getParameter<double>("dupMaxDeltaR2")),
        dupPtDifference_(iConfig.getParameter<double>("dupPtDifference")),
        trackTags_(iConfig.getParameter<std::vector<edm::InputTag>>("inputTracks")),
        outputTracks_(produces()) {
    for (const auto& it : trackTags_) {
      trackTokens_.push_back(consumes(it));
    }

    assert(trackTags_.size() <= ::mergerKernels::maxTrackSoACollections);

    if (trackTags_.empty()) {
      throw cms::Exception("TracksSoAMerger - Inputs") << "No input TkSoA collections provided";
    }
    if (trackTags_.size() > ::mergerKernels::maxTrackSoACollections) {
      throw cms::Exception("TracksSoAMerger - Inputs")
          << "Too many input TkSoA collections provided.\n Maximum allowed is "
          << ::mergerKernels::maxTrackSoACollections;
    }

    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("TracksSoAMerger - PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("TracksSoAMerger - PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }
  }

  void TracksSoAMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::vector<edm::InputTag>>(
        "inputTracks", {edm::InputTag("pixelTracksHighPtAlpaka"), edm::InputTag("pixelTracksLowPtAlpaka")});
    desc.add<std::string>("minQuality", "highPurity");
    desc.add<double>("matchFraction", 0.0);
    desc.add<int>("minHitsForDuplicate", 3);
    desc.add<double>("dupNSigma2", 3.0);
    desc.add<double>("dupMaxDeltaR2", 0.0004);
    desc.add<double>("dupPtDifference", 0.1);

    descriptions.addWithDefaultLabel(desc);
  }

  void TracksSoAMerger::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto queue = iEvent.queue();

    std::vector<const Tracks*> trackSoAs;
    trackSoAs.resize(trackTokens_.size());

    auto maxTracks = 0;
#ifdef GPU_DEBUG
    std::cout << "TracksSoAMerger::acquire: nCollections_: " << trackTokens_.size() << std::endl;
#endif

    for (auto i = 0u; i < trackTokens_.size(); ++i) {
      auto const& aux = iEvent.get(trackTokens_[i]);
      trackSoAs[i] = &aux;
      maxTracks += aux.view().tracks().metadata().size();
#ifdef GPU_DEBUG
      std::cout << "TracksSoAMerger::acquire: trackSoAs[" << i << "]: " << trackTags_[i]
                << ", nTracks: " << aux.view().tracks().metadata().size() << std::endl;
#endif
    }

    TracksMultiView tracksViews(
        trackSoAs, [](const Tracks* tracks) -> auto { return TracksConstView(tracks->const_view().tracks()); });
    TrackHitsMultiView hitsViews(
        trackSoAs, [](const Tracks* tracks) -> auto { return TrackHitsConstView(tracks->const_view().trackHits()); });

    if (tracksViews.numViews() != hitsViews.numViews())
      throw cms::Exception("TracksSoAMerger::acquire: numViews()")
          << "Number of track views does not match number of hit views\n";

    bool doSameHitsDuplicates = (dupMinHits_ > 0) || (matchFraction_ > 0.0);  //these are min
    bool doParmsDupRejection =
        (dupNSigma2_ > 0.0) && (dupMaxDeltaR2_ > 0.0) && (dupPtDifference_ > 0.0);  //these are max

    AlgoParams params{doSameHitsDuplicates,
                      doParmsDupRejection,
                      minQuality_,
                      maxTracks,
                      dupMinHits_,
                      matchFraction_,
                      dupNSigma2_,
                      dupMaxDeltaR2_,
                      dupPtDifference_};
    Algo deviceAlgo_(queue, params);

    if (maxTracks > 0) {
      tracks_d_ = deviceAlgo_.makeMergedTracks(queue, tracksViews, hitsViews);
    } else {
      tracks_d_ = Tracks(queue, 0, 0);
    }
  }

  void TracksSoAMerger::produce(device::Event& iEvent, const device::EventSetup& es) {
    iEvent.emplace(outputTracks_, std::move(*tracks_d_));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksSoAMerger);
