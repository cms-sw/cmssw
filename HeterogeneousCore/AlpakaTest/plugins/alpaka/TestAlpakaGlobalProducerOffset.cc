#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class is primarily for consuming AlpakaESTestDataEDevice
   * ESProduct. It also demonstrates a proof-of-concept for
   * backend-specific configuration parameters.
   */
  class TestAlpakaGlobalProducerOffset : public global::EDProducer<> {
  public:
    TestAlpakaGlobalProducerOffset(edm::ParameterSet const& config)
        : esToken_(esConsumes()),
          deviceToken_{produces()},
          x_(config.getParameter<edm::ParameterSet>("xvalue").getParameter<double>(
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE))) {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      auto const& esData = iSetup.getData(esToken_);

      portabletest::TestDeviceCollection deviceProduct{esData->metadata().size(), alpaka::getDev(iEvent.queue())};

      // run the algorithm, potentially asynchronously
      algo_.fill(iEvent.queue(), deviceProduct, x_);

      iEvent.emplace(deviceToken_, std::move(deviceProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;

      // TODO: if this becomes an accepted pattern, we could look into
      // simplifying the definition (also this use case has additional
      // constraints not in a general PSet)
      edm::ParameterSetDescription psetX;
      psetX.add<double>("alpaka_serial_sync", 0.);
      psetX.add<double>("alpaka_cuda_async", 0.);
      desc.add("xvalue", psetX);

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    device::ESGetToken<AlpakaESTestDataADevice, AlpakaESTestRecordA> const esToken_;
    device::EDPutToken<portabletest::TestDeviceCollection> const deviceToken_;

    TestAlgo const algo_{};

    double const x_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaGlobalProducerOffset);
