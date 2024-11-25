#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestSoA.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class is the equivalent of TesAlpakaESProducerA.cc
   * for PortableHostMultiCollection. It consumes a standard
   * host ESProduct and converts the data into PortableHostMultiCollection, and
   * implicitly transfers the data product to device
   */
  class TestAlpakaESProducerAMulti : public ESProducer {
  public:
    TestAlpakaESProducerAMulti(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      token_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<AlpakaESTestDataACMultiHost> produce(AlpakaESTestRecordA const& iRecord) {
      auto const& input = iRecord.get(token_);

      int const sizeA = 10;
      int const sizeC = 100;
      // TODO: pinned allocation?
      // TODO: cached allocation?
      AlpakaESTestDataACMultiHost product({{sizeA, sizeC}}, cms::alpakatools::host());
      auto viewA = product.view<
          cms::alpakatest::AlpakaESTestSoAA>();  // this template is not really needed as this is fhe first layout
      auto viewC = product.view<cms::alpakatest::AlpakaESTestSoAC>();

      for (int i = 0; i < sizeA; ++i) {
        viewA[i].z() = input.value() - i;
      }

      for (int i = 0; i < sizeC; ++i) {
        viewC[i].x() = input.value() + i;
      }

      return product;
    }

  private:
    edm::ESGetToken<cms::alpakatest::ESTestDataA, AlpakaESTestRecordA> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESProducerAMulti);
