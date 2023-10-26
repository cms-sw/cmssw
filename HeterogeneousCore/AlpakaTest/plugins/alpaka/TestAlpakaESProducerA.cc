#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates an ESProducer that uses the
   * PortableCollection-based data model, and that consumes a standard
   * host ESProduct and converts the data into PortableCollection, and
   * implicitly transfers the data product to device
   */
  class TestAlpakaESProducerA : public ESProducer {
  public:
    TestAlpakaESProducerA(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      token_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<AlpakaESTestDataAHost> produce(AlpakaESTestRecordA const& iRecord) {
      auto const& input = iRecord.get(token_);

      int const size = 10;
      // TODO: pinned allocation?
      // TODO: cached allocation?
      auto product = std::make_unique<AlpakaESTestDataAHost>(size, cms::alpakatools::host());
      for (int i = 0; i < size; ++i) {
        product->view()[i].z() = input.value() - i;
      }
      return product;
    }

  private:
    edm::ESGetToken<cms::alpakatest::ESTestDataA, AlpakaESTestRecordA> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESProducerA);
