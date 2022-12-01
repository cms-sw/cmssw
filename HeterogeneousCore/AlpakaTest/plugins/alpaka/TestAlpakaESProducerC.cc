#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates and ESProducer on the data model 3 that
   * - consumes a standard host ESProduct and converts the data into an Alpaka buffer
   * - transfers the buffer contents to the device of the backend
   */
  class TestAlpakaESProducerC : public ESProducer {
  public:
    TestAlpakaESProducerC(edm::ParameterSet const& iConfig) {
      {
        auto cc = setWhatProduced(this, &TestAlpakaESProducerC::produceHost);
        token_ = cc.consumes();
      }
      {
        auto cc = setWhatProduced(this, &TestAlpakaESProducerC::produceDevice);
        hostToken_ = cc.consumes();
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<AlpakaESTestDataCHost> produceHost(AlpakaESTestRecordC const& iRecord) {
      auto const& input = iRecord.get(token_);

      int const size = 5;
      // TODO: pinned allocation?
      // TODO: cached allocation?
      AlpakaESTestDataCHost product(size, cms::alpakatools::host());
      for (int i = 0; i < size; ++i) {
        product.view()[i].x() = input.value() - i;
      }
      return product;
    }

    // TODO: in principle in this model the transfer to device could be automated
    std::optional<AlpakaESTestDataCDevice> produceDevice(device::Record<AlpakaESTestRecordC> const& iRecord) {
      auto hostHandle = iRecord.getTransientHandle(hostToken_);
      auto const& hostProduct = *hostHandle;
      AlpakaESTestDataCDevice deviceProduct(hostProduct->metadata().size(), iRecord.queue());
      alpaka::memcpy(iRecord.queue(), deviceProduct.buffer(), hostProduct.buffer());

      return deviceProduct;
    }

  private:
    edm::ESGetToken<cms::alpakatest::ESTestDataC, AlpakaESTestRecordC> token_;
    edm::ESGetToken<AlpakaESTestDataCHost, AlpakaESTestRecordC> hostToken_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESProducerC);
