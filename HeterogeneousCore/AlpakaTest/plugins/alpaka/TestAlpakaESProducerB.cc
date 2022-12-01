#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates and ESProducer on the data model 2 that
   * - consumes a standard host ESProduct and converts the data into an Alpaka buffer
   * - transfers the buffer contents to the device of the backend
   */
  class TestAlpakaESProducerB : public ESProducer {
  public:
    TestAlpakaESProducerB(edm::ParameterSet const& iConfig) {
      {
        auto cc = setWhatProduced(this, &TestAlpakaESProducerB::produceHost);
        token_ = cc.consumes();
      }
      {
        auto cc = setWhatProduced(this, &TestAlpakaESProducerB::produceDevice);
        hostToken_ = cc.consumes();
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<cms::alpakatest::AlpakaESTestDataB<DevHost>> produceHost(AlpakaESTestRecordB const& iRecord) {
      auto const& input = iRecord.get(token_);

      int const size = 5;
      // TODO: cached allocation?
      auto buffer = cms::alpakatools::make_host_buffer<int[], Platform>(size);
      for (int i = 0; i < size; ++i) {
        buffer[i] = i + input.value();
      }
      return cms::alpakatest::AlpakaESTestDataB<DevHost>(std::move(buffer));
    }

    // TODO: in principle in this model the transfer to device could be automated
    std::optional<cms::alpakatest::AlpakaESTestDataB<Device>> produceDevice(
        device::Record<AlpakaESTestRecordB> const& iRecord) {
      auto hostHandle = iRecord.getTransientHandle(hostToken_);
      auto buffer = cms::alpakatools::make_device_buffer<int[]>(iRecord.queue(), hostHandle->size());
      alpaka::memcpy(iRecord.queue(), buffer, hostHandle->buffer());
      return cms::alpakatest::AlpakaESTestDataB<Device>(std::move(buffer));
    }

  private:
    edm::ESGetToken<cms::alpakatest::ESTestDataB, AlpakaESTestRecordB> token_;
    edm::ESGetToken<cms::alpakatest::AlpakaESTestDataB<DevHost>, AlpakaESTestRecordB> hostToken_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESProducerB);
