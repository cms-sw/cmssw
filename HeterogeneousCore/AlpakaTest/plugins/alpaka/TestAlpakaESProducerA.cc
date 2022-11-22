#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates and ESProducer on the data model 1 that
   * - consumes a standard host ESProduct and converts the data into an Alpaka buffer
   * - transfers the buffer contents to the device of the backend
   */
  class TestAlpakaESProducerA : public ESProducer {
  public:
    class HostBuffer {
    public:
      using Buffer = cms::alpakatools::device_buffer<DevHost, int[]>;
      using ConstBuffer = cms::alpakatools::const_device_buffer<DevHost, int[]>;

      HostBuffer(Buffer buffer) : buffer_(std::move(buffer)) {}

      ConstBuffer buffer() const { return buffer_; }

    private:
      Buffer buffer_;
    };

    TestAlpakaESProducerA(edm::ParameterSet const& iConfig) {
      {
        auto cc = setWhatProduced(this, &TestAlpakaESProducerA::produceHost);
        token_ = cc.consumes();
      }
      {
        auto cc = setWhatProduced(this, &TestAlpakaESProducerA::produceDevice);
        hostToken_ = cc.consumes();
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<HostBuffer> produceHost(AlpakaESTestRecordA const& iRecord) {
      auto const& input = iRecord.get(token_);

      int const size = 10;
      // TODO: cached allocation?
      auto buffer = cms::alpakatools::make_host_buffer<int[], Platform>(size);
      for (int i = 0; i < size; ++i) {
        buffer[i] = i * input.value();
      }
      return HostBuffer(std::move(buffer));
    }

    std::optional<AlpakaESTestDataA> produceDevice(device::Record<AlpakaESTestRecordA> const& iRecord) {
      auto hostHandle = iRecord.getTransientHandle(hostToken_);
      // TODO: In principle associating the allocation to a queue is
      // incorrect. Framework will keep the memory alive until the IOV
      // ends. By that point all asynchronous activity using that
      // memory has finished, and the memory could be marked as "free"
      // in the allocator already by the host-side release of the
      // memory. There could also be other, independent asynchronous
      // activity going on that uses the same queue (since we don't
      // retain the queue here), and at the time of host-side release
      // the device-side release gets associated to the complemention
      // of that activity (which has nothing to do with the memory here).
      auto buffer =
          cms::alpakatools::make_device_buffer<int[]>(iRecord.queue(), alpaka::getExtentProduct(hostHandle->buffer()));
      alpaka::memcpy(iRecord.queue(), buffer, hostHandle->buffer());
      return AlpakaESTestDataA(std::move(buffer));
    }

  private:
    edm::ESGetToken<cms::alpakatest::ESTestDataA, AlpakaESTestRecordA> token_;
    edm::ESGetToken<HostBuffer, AlpakaESTestRecordA> hostToken_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESProducerA);
// for the temporary host data
TYPELOOKUP_DATA_REG(ALPAKA_ACCELERATOR_NAMESPACE::TestAlpakaESProducerA::HostBuffer);
