#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class tests various ways of a device ESProduct being missing
   */
  class TestAlpakaGlobalProducerNullES : public global::EDProducer<> {
  public:
    TestAlpakaGlobalProducerNullES(edm::ParameterSet const& config)
        : esTokenA_(esConsumes(config.getParameter<edm::ESInputTag>("eventSetupSource"))),
          esTokenC_(esConsumes(config.getParameter<edm::ESInputTag>("eventSetupSource"))),
          esTokenCNotExist_(esConsumes(edm::ESInputTag("", "doesNotExist"))) {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      bool threw = false;
      try {
        [[maybe_unused]] auto handleA = iSetup.getHandle(esTokenA_);
      } catch (cms::Exception& e) {
        threw = true;
      }
      if (not threw) {
        throw cms::Exception("Assert") << "Getting AlpakaESTestDataADevice ESProduct did not throw";
      }

      threw = false;
      try {
        [[maybe_unused]] auto const& prodC = iSetup.getData(esTokenC_);
      } catch (cms::Exception& e) {
        threw = true;
      }
      if (not threw) {
        throw cms::Exception("Assert") << "Getting AlpakaESTestDataCDevice ESProduct did not throw";
      }

      auto handleC = iSetup.getHandle(esTokenCNotExist_);
      if (handleC.isValid()) {
        throw cms::Exception("Assert") << "Getting non-existing AlpakaESTestDataCDevice succeeded, should have failed";
      }
      threw = false;
      try {
        [[maybe_unused]] auto const& prodC = *handleC;
      } catch (cms::Exception& e) {
        threw = true;
      }
      if (not threw) {
        throw cms::Exception("Assert")
            << "De-referencing ESHandle of non-existentAlpakaESTestDataADevice did not throw";
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("eventSetupSource", edm::ESInputTag{});
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<AlpakaESTestDataADevice, AlpakaESTestRecordA> esTokenA_;
    const device::ESGetToken<AlpakaESTestDataCDevice, AlpakaESTestRecordA> esTokenC_;
    const device::ESGetToken<AlpakaESTestDataCDevice, AlpakaESTestRecordA> esTokenCNotExist_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestAlpakaGlobalProducerNullES);
