#include <alpaka/alpaka.hpp>

#include <numeric>

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
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
    std::vector<device::EDGetToken<reco::TracksSoACollection>> inputTkSoATokenV_;
    std::vector<edm::InputTag> inputTkSoATagV_;
    const device::EDPutToken<reco::TracksSoACollection> outputTkSoAToken_;

    // input collections
    ::mergerKernels::InputTracks allTrackView_;
    int nCollections_ = 0;

    // output tracks
    std::optional<reco::TracksSoACollection> tracks_d_;
    int maxTracks_ = 0;
  };

  TracksSoAMerger::TracksSoAMerger(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        matchFraction_(iConfig.getParameter<double>("matchFraction")),
        dupMinHits_(iConfig.getParameter<int>("minHitsForDuplicate")),
        dupNSigma2_(iConfig.getParameter<double>("dupNSigma2")),
        dupMaxDeltaR2_(iConfig.getParameter<double>("dupMaxDeltaR2")),
        dupPtDifference_(iConfig.getParameter<double>("dupPtDifference")),
        inputTkSoATagV_(iConfig.getParameter<std::vector<edm::InputTag>>("inputTkSoAs")),
        outputTkSoAToken_(produces()) {
    for (const auto& it : inputTkSoATagV_) {
      inputTkSoATokenV_.push_back(consumes(it));
    }

    assert(inputTkSoATagV_.size() <= ::mergerKernels::maxTrackSoACollections);
    nCollections_ = inputTkSoATagV_.size();
    allTrackView_.nInputs = inputTkSoATagV_.size();

    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }
  }

  void TracksSoAMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::vector<edm::InputTag>>(
        "inputTkSoAs", {edm::InputTag("pixelTracksHighPtAlpaka"), edm::InputTag("pixelTracksLowPtAlpaka")});
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

    std::vector<const reco::TracksSoACollection*> inputTkSoAs;
    inputTkSoAs.resize(inputTkSoATokenV_.size());

    maxTracks_ = 0;
#ifdef GPU_DEBUG
    std::cout << "TracksSoAMerger::acquire: nCollections_: " << nCollections_ << std::endl;
#endif
    for (int i = 0; i < nCollections_; ++i) {
      auto const& aux = iEvent.get(inputTkSoATokenV_[i]);
      inputTkSoAs[i] = &aux;
      allTrackView_.views[i] = aux.view().tracks();
      maxTracks_ += aux.view().tracks().metadata().size();
      allTrackView_.hitViews[i] = aux.view().trackHits();
#ifdef GPU_DEBUG
      std::cout << "TracksSoAMerger::acquire: inputTkSoAs[" << i << "]: " << inputTkSoATagV_[i]
                << ", nTracks: " << aux.view().tracks().metadata().size() << std::endl;
#endif
    }

    allTrackView_.nTracks = maxTracks_;
    bool doSameHitsDuplicates = (dupMinHits_ > 0) || (matchFraction_ > 0.0);  //these are min
    bool doParmsDupRejection =
        (dupNSigma2_ > 0.0) && (dupMaxDeltaR2_ > 0.0) && (dupPtDifference_ > 0.0);  //these are max

    AlgoParams params{doSameHitsDuplicates,
                      doParmsDupRejection,
                      minQuality_,
                      maxTracks_,
                      dupMinHits_,
                      matchFraction_,
                      dupNSigma2_,
                      dupMaxDeltaR2_,
                      dupPtDifference_};
    Algo deviceAlgo_(params, queue);

    if (maxTracks_ > 0) {
      tracks_d_ = deviceAlgo_.makeMergedTracks(queue, allTrackView_);  //TODO: better constructor
    } else {
      tracks_d_ = reco::TracksSoACollection(queue, 0, 0);
    }
  }

  void TracksSoAMerger::produce(device::Event& iEvent, const device::EventSetup& es) {
    iEvent.emplace(outputTkSoAToken_, std::move(*tracks_d_));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksSoAMerger);
