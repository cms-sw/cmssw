#include <alpaka/alpaka.hpp>

#include "RecoTracker/LSTCore/interface/alpaka/LST.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/LST/interface/LSTOutput.h"
#include "RecoTracker/LST/interface/LSTPhase2OTHitsInput.h"
#include "RecoTracker/LST/interface/LSTPixelSeedInput.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTProducer : public stream::SynchronizingEDProducer<> {
  public:
    LSTProducer(edm::ParameterSet const& config)
        : lstPixelSeedInputToken_{consumes(config.getParameter<edm::InputTag>("pixelSeedInput"))},
          lstPhase2OTHitsInputToken_{consumes(config.getParameter<edm::InputTag>("phase2OTHitsInput"))},
          lstESToken_{esConsumes()},
          verbose_(config.getParameter<bool>("verbose")),
          nopLSDupClean_(config.getParameter<bool>("nopLSDupClean")),
          tcpLSTriplets_(config.getParameter<bool>("tcpLSTriplets")),
          lstOutputToken_{produces()} {}

    void acquire(device::Event const& event, device::EventSetup const& setup) override {
      // Inputs
      auto const& pixelSeeds = event.get(lstPixelSeedInputToken_);
      auto const& phase2OTHits = event.get(lstPhase2OTHitsInputToken_);

      auto const& lstESDeviceData = setup.getData(lstESToken_);

      lst_.run(event.queue(),
               verbose_,
               &lstESDeviceData,
               pixelSeeds.px(),
               pixelSeeds.py(),
               pixelSeeds.pz(),
               pixelSeeds.dxy(),
               pixelSeeds.dz(),
               pixelSeeds.ptErr(),
               pixelSeeds.etaErr(),
               pixelSeeds.stateTrajGlbX(),
               pixelSeeds.stateTrajGlbY(),
               pixelSeeds.stateTrajGlbZ(),
               pixelSeeds.stateTrajGlbPx(),
               pixelSeeds.stateTrajGlbPy(),
               pixelSeeds.stateTrajGlbPz(),
               pixelSeeds.q(),
               pixelSeeds.hitIdx(),
               phase2OTHits.detId(),
               phase2OTHits.x(),
               phase2OTHits.y(),
               phase2OTHits.z(),
               nopLSDupClean_,
               tcpLSTriplets_);
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      // Output
      LSTOutput lstOutput(lst_.hits(), lst_.len(), lst_.seedIdx(), lst_.trackCandidateType());
      event.emplace(lstOutputToken_, std::move(lstOutput));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("pixelSeedInput", edm::InputTag{"lstPixelSeedInputProducer"});
      desc.add<edm::InputTag>("phase2OTHitsInput", edm::InputTag{"lstPhase2OTHitsInputProducer"});
      desc.add<bool>("verbose", false);
      desc.add<bool>("nopLSDupClean", false);
      desc.add<bool>("tcpLSTriplets", false);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<LSTPixelSeedInput> lstPixelSeedInputToken_;
    edm::EDGetTokenT<LSTPhase2OTHitsInput> lstPhase2OTHitsInputToken_;
    device::ESGetToken<lst::LSTESData<Device>, TrackerRecoGeometryRecord> lstESToken_;
    const bool verbose_, nopLSDupClean_, tcpLSTriplets_;
    edm::EDPutTokenT<LSTOutput> lstOutputToken_;

    lst::LST lst_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LSTProducer);
