#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/TestHostOnlyHelperClass.h"

#include "TestHelperClass.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a stream EDProducer that
   * - uses a helper class (need to use edm::ConsumesCollector), that
   *   - consumes a device EDProduct
   *   - consumes a host ESProduct
   *   - consumes a device ESProduct
   * - uses a non-Alpaka-aware helper class (need to use edm::ConsumesCollector), that
   *   - consumes a host EDProduct
   *   - consumes a host ESProduct
   * - consumes a device ESProduct
   * - produces a host EDProduct
   * - synchronizes in a non-blocking way with the ExternalWork module
   *   ability (via the SynchronizingEDProcucer base class)
   */
  class TestAlpakaStreamSynchronizingProducer : public stream::SynchronizingEDProducer<> {
  public:
    TestAlpakaStreamSynchronizingProducer(edm::ParameterSet const& iConfig)
        : esTokenDevice_(esConsumes()),
          putToken_{produces()},
          helper_{iConfig, consumesCollector()},
          hostHelper_{iConfig, consumesCollector()},
          expectedInt_{iConfig.getParameter<int>("expectedInt")} {}

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      [[maybe_unused]] auto const& esData = iSetup.getData(esTokenDevice_);

      int const value = hostHelper_.run(iEvent, iSetup);
      if (value != expectedInt_) {
        throw cms::Exception("Assert") << "Expected value " << expectedInt_ << ", but got " << value;
      }

      helper_.makeAsync(iEvent, iSetup);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      iEvent.emplace(putToken_, helper_.moveFrom());
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      TestHelperClass::fillPSetDescription(desc);
      cms::alpakatest::TestHostOnlyHelperClass::fillPSetDescription(desc);
      desc.add<int>("expectedInt");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<AlpakaESTestDataDDevice, AlpakaESTestRecordD> esTokenDevice_;
    const edm::EDPutTokenT<portabletest::TestHostCollection> putToken_;

    TestHelperClass helper_;
    cms::alpakatest::TestHostOnlyHelperClass const hostHelper_;
    int const expectedInt_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaStreamSynchronizingProducer);
