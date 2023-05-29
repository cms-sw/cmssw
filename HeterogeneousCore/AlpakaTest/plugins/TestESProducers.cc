#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"

namespace cms::alpakatest {
  template <typename TProduct, typename TRecord>
  class TestESProducerT : public edm::ESProducer {
  public:
    TestESProducerT(edm::ParameterSet const& iConfig) : value_(iConfig.getParameter<int>("value")) {
      setWhatProduced(this);
    }

    std::optional<TProduct> produce(TRecord const& iRecord) { return TProduct(value_); }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("value");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    int const value_;
  };

  using TestESProducerA = TestESProducerT<cms::alpakatest::ESTestDataA, AlpakaESTestRecordA>;
  using TestESProducerB = TestESProducerT<cms::alpakatest::ESTestDataB, AlpakaESTestRecordB>;
  using TestESProducerC = TestESProducerT<cms::alpakatest::ESTestDataC, AlpakaESTestRecordC>;
}  // namespace cms::alpakatest

DEFINE_FWK_EVENTSETUP_MODULE(cms::alpakatest::TestESProducerA);
DEFINE_FWK_EVENTSETUP_MODULE(cms::alpakatest::TestESProducerB);
DEFINE_FWK_EVENTSETUP_MODULE(cms::alpakatest::TestESProducerC);
