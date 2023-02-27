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
   * consumes a standard host ESProduct and converts the data into an
   * Alpaka buffer that is then moved into an object of a class that
   * is templated over the device type, and implicitly transfers the
   * data product to device
   *
   * This class also tests the explicit label for ESProducts works
   */
  class TestAlpakaESProducerB : public ESProducer {
  public:
    TestAlpakaESProducerB(edm::ParameterSet const& iConfig) {
      auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("explicitLabel"));
      token_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("explicitLabel", std::string{});
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<cms::alpakatest::AlpakaESTestDataB<DevHost>> produce(AlpakaESTestRecordB const& iRecord) {
      auto const& input = iRecord.get(token_);

      int const size = 5;
      // TODO: cached allocation?
      auto buffer = cms::alpakatools::make_host_buffer<int[], Platform>(size);
      for (int i = 0; i < size; ++i) {
        buffer[i] = i + input.value();
      }
      return cms::alpakatest::AlpakaESTestDataB<DevHost>(std::move(buffer));
    }

  private:
    edm::ESGetToken<cms::alpakatest::ESTestDataB, AlpakaESTestRecordB> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESProducerB);
