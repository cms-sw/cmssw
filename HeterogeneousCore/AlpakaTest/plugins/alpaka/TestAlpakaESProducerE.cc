#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

#include <cmath>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates an ESProducer that uses the
   * PortableCollection-based data model, and that consumes a standard
   * host ESProduct and converts the data into PortableCollection, and
   * implicitly transfers the data product to device
   */
  class TestAlpakaESProducerE : public ESProducer {
  public:
    TestAlpakaESProducerE(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      token_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<AlpakaESTestDataEHost> produce(AlpakaESTestRecordC const& iRecord) {
      auto const& input = iRecord.get(token_);

      int const edatasize = 2;
      AlpakaESTestDataEHost::EDataCollection data(edatasize, cms::alpakatools::host());
      for (int i = 0; i < edatasize; ++i) {
        data.view()[i].val2() = i * 10 + 1;
      }

      int const esize = 5;
      // TODO: pinned allocation?
      // TODO: cached allocation?
      AlpakaESTestDataEHost::ECollection e(esize, cms::alpakatools::host());
      for (int i = 0; i < esize; ++i) {
        e.view()[i].val() = std::abs(input.value()) + i * 2;
        e.view()[i].ind() = i % edatasize;
      }
      return AlpakaESTestDataEHost(std::move(e), std::move(data));
    }

  private:
    edm::ESGetToken<cms::alpakatest::ESTestDataC, AlpakaESTestRecordC> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TestAlpakaESProducerE);
