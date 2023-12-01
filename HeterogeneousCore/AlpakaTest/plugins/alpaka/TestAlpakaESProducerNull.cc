#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates an ESProducer that
   * - produces a null in a host produce function
   * - produces a null in a device produce function
   */
  class TestAlpakaESProducerNull : public ESProducer {
  public:
    TestAlpakaESProducerNull(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      setWhatProduced(this, &TestAlpakaESProducerNull::produceHost);
      setWhatProduced(this, &TestAlpakaESProducerNull::produceDevice);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<AlpakaESTestDataAHost> produceHost(AlpakaESTestRecordA const& iRecord) { return {}; }

    std::unique_ptr<AlpakaESTestDataCDevice> produceDevice(device::Record<AlpakaESTestRecordD> const& iRecord) {
      return {};
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESProducerNull);
