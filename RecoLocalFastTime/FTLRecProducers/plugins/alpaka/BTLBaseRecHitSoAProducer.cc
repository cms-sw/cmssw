#include <utility>

#include "DataFormats/FTLRecHitSoA/interface/alpaka/BTLBaseRecHitDeviceCollection.h"
#include "DataFormats/FTLDigiSoA/interface/BTLDigiHostCollection.h"
#include "DataFormats/FTLDigiSoA/interface/alpaka/BTLDigiDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "BTLBaseRecHitSoAProducerAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit {

  using namespace ::btlrechit;

  class BTLBaseRecHitSoAProducer : public stream::EDProducer<> {
  public:
    // constructor
    BTLBaseRecHitSoAProducer(edm::ParameterSet const& config)
        : EDProducer<>(config),
          digi_(consumes<::btldigi::BTLDigiHostCollection>(config.getParameter<edm::InputTag>("digi"))),
          uncalibrh_{produces()},
          adcBitSaturation_(config.getParameter<uint32_t>("adcBitSaturation")),
          tclock_(config.getParameter<double>("tclock")),
          tdcCalParams_(config.getParameter<std::vector<double>>("tdcCalParams")),
          qdcCalParams_(config.getParameter<std::vector<double>>("qdcCalParams")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("digi");
      desc.add<uint32_t>("adcBitSaturation");
      desc.add<double>("tclock");
      desc.add<std::vector<double>>("tdcCalParams");
      desc.add<std::vector<double>>("qdcCalParams");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& event, device::EventSetup const& setup) override {
      // NB should be inserted a method to retrieve calibrations, now they are fixed to default values
      // Get the digi from the Event.
      auto const& hostDigi = event.get(digi_);  // SoA DIGI stored in the event
      auto const N = hostDigi.const_view().metadata().size();

      // Copy input to device for GPU inference
      btldigi::BTLDigiDeviceCollection deviceDigi(event.queue(), N);
      alpaka::memcpy(event.queue(), deviceDigi.buffer(), hostDigi.buffer());

      // Allocate a new SoA for the uncalibrh jets. // same number of elements we have in input
      BTLBaseRecHitDeviceCollection uncalibrh(event.queue(), N);

      // Apply the corrections and fill the new SoA. // these launch the kernel, and will run on gpu async
      std::array<double, 4> tdcCalParamsArray_;
      std::array<double, 10> qdcCalParamsArray_;
      std::copy_n(tdcCalParams_.begin(), 4, tdcCalParamsArray_.begin());
      std::copy_n(qdcCalParams_.begin(), 10, qdcCalParamsArray_.begin());
      BTLBaseRecHitSoAProducerAlgo::fromDigiToBase(event.queue(),
                                                   deviceDigi.view(),
                                                   uncalibrh.view(),
                                                   adcBitSaturation_,
                                                   tclock_,
                                                   tdcCalParamsArray_,
                                                   qdcCalParamsArray_);

      // Move the SoA with the uncalibrh jets into the Event.
      event.emplace(uncalibrh_, std::move(uncalibrh));
    }

  private:
    const edm::EDGetTokenT<::btldigi::BTLDigiHostCollection> digi_;
    const device::EDPutToken<BTLBaseRecHitDeviceCollection> uncalibrh_;
    const uint32_t adcBitSaturation_;
    const double tclock_;
    const std::vector<double> tdcCalParams_;
    const std::vector<double> qdcCalParams_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(btlrechit::BTLBaseRecHitSoAProducer);
