#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TestHelperClass.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class demonstrates a stream EDProducer that
   * - uses a helper class (need to use edm::ConsumesCollector), that
   *   - consumes a device EDProduct
   *   - consumes a host ESProduct
   *   - consumes a device ESProduct
   * - consumes a device ESProduct
   * - produces a host EDProduct
   * - synchronizes in a non-blocking way with the ExternalWork module
   *   ability (via the SynchronizingEDProcucer base class)
   */
  class TestAlpakaStreamSynchronizingProducer : public stream::SynchronizingEDProducer<> {
  public:
    TestAlpakaStreamSynchronizingProducer(edm::ParameterSet const& iConfig)
        : esTokenDevice_(esConsumes()), putToken_{produces()}, helper_{iConfig, consumesCollector()} {}

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      [[maybe_unused]] auto const& esData = iSetup.getData(esTokenDevice_);

      helper_.makeAsync(iEvent, iSetup);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      iEvent.emplace(putToken_, helper_.moveFrom());
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<AlpakaESTestDataDDevice, AlpakaESTestRecordD> esTokenDevice_;
    const edm::EDPutTokenT<portabletest::TestHostCollection> putToken_;

    TestHelperClass helper_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaStreamSynchronizingProducer);
