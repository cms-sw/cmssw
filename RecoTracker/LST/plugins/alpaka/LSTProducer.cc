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

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTProducer : public stream::SynchronizingEDProducer<> {
  public:
    LSTProducer(edm::ParameterSet const& config)
        : SynchronizingEDProducer(config),
          lstInputToken_{consumes(config.getParameter<edm::InputTag>("lstInput"))},
          lstESToken_{esConsumes(edm::ESInputTag("", config.getParameter<std::string>("ptCutLabel")))},
          verbose_(config.getParameter<bool>("verbose")),
          ptCut_(config.getParameter<double>("ptCut")),
          nopLSDupClean_(config.getParameter<bool>("nopLSDupClean")),
          tcpLSTriplets_(config.getParameter<bool>("tcpLSTriplets")),
          lstOutputToken_{produces()} {}

    void acquire(device::Event const& event, device::EventSetup const& setup) override {
      // Inputs
      auto const& lstInputDC = event.get(lstInputToken_);

      auto const& lstESDeviceData = setup.getData(lstESToken_);

      lst_.run(event.queue(),
               verbose_,
               static_cast<float>(ptCut_),
               &lstESDeviceData,
               &lstInputDC,
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
      desc.add<edm::InputTag>("lstInput", edm::InputTag{"lstInputProducer"});
      desc.add<bool>("verbose", false);
      desc.add<double>("ptCut", 0.8);
      desc.add<std::string>("ptCutLabel", "0.8");
      desc.add<bool>("nopLSDupClean", false);
      desc.add<bool>("tcpLSTriplets", false);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    device::EDGetToken<lst::LSTInputDeviceCollection> lstInputToken_;
    device::ESGetToken<lst::LSTESData<Device>, TrackerRecoGeometryRecord> lstESToken_;
    const bool verbose_;
    const double ptCut_;
    const bool nopLSDupClean_;
    const bool tcpLSTriplets_;
    edm::EDPutTokenT<LSTOutput> lstOutputToken_;

    lst::LST lst_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LSTProducer);
