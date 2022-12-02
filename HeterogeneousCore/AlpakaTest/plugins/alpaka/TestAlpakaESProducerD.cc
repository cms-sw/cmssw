#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

#include "testESAlgoAsync.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates an ESProducer that
   * - consumes device ESProducts (that were produced on different Records)
   * - calls a kernel on the queue
   */
  class TestAlpakaESProducerD : public ESProducer {
  public:
    TestAlpakaESProducerD(edm::ParameterSet const& iConfig) {
      auto cc = setWhatProduced(this);
      tokenA_ = cc.consumes();
      tokenB_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<AlpakaESTestDataDDevice> produce(device::Record<AlpakaESTestRecordD> const& iRecord) {
      auto const& dataA = iRecord.get(tokenA_);
      auto const& dataB = iRecord.get(tokenB_);

      return testESAlgoAsync(iRecord.queue(), dataA, dataB);
    }

  private:
    device::ESGetToken<AlpakaESTestDataADevice, AlpakaESTestRecordA> tokenA_;
    device::ESGetToken<cms::alpakatest::AlpakaESTestDataB<Device>, AlpakaESTestRecordB> tokenB_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESProducerD);
