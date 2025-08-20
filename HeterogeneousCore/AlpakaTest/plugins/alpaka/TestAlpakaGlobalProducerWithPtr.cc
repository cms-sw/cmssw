#include "DataFormats/PortableTestObjects/interface/TestProductWithPtr.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

#include "testPtrAlgoAsync.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class is part of testing CopyToHost<T>::postCopy().
   */
  class TestAlpakaGlobalProducerWithPtr : public global::EDProducer<> {
  public:
    TestAlpakaGlobalProducerWithPtr(edm::ParameterSet const& config)
        : EDProducer<>(config), token_{produces()}, size_{config.getParameter<int>("size")} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<int>("size");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      iEvent.emplace(token_, testPtrAlgoAsync(iEvent.queue(), size_));
    }

  private:
    const device::EDPutToken<portabletest::TestProductWithPtr<Device>> token_;
    const int32_t size_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaGlobalProducerWithPtr);
