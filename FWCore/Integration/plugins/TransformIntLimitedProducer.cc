#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/limited/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edmtest {
  class TransformIntLimitedProducer : public edm::limited::EDProducer<edm::Transformer> {
  public:
    TransformIntLimitedProducer(edm::ParameterSet const& iPSet)
        : edm::limited::EDProducerBase(iPSet),
          edm::limited::EDProducer<edm::Transformer>(iPSet),
          getToken_(consumes(iPSet.getParameter<edm::InputTag>("get"))),
          offset_(iPSet.getParameter<unsigned int>("offset")),
          transformOffset_(iPSet.getParameter<unsigned int>("transformOffset")) {
      putToken_ = produces<IntProduct>();
      bool check = iPSet.getUntrackedParameter<bool>("checkTransformNotCalled");
      registerTransform(
          putToken_,
          [offset = transformOffset_, check](edm::StreamID, auto const& iFrom) {
            if (check) {
              throw cms::Exception("TransformShouldNotBeCalled");
            }
            return IntProduct(iFrom.value + offset);
          },
          "transform");
    }

    void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
      iEvent.emplace(putToken_, iEvent.get(getToken_).value + offset_);
    }
    static void fillDescriptions(edm::ConfigurationDescriptions& desc) {
      edm::ParameterSetDescription pset;
      pset.add<edm::InputTag>("get");
      pset.add<unsigned int>("offset", 0);
      pset.add<unsigned int>("transformOffset", 1);
      pset.addUntracked<bool>("checkTransformNotCalled", false);
      pset.addUntracked<unsigned int>("concurrencyLimit", 1);

      desc.addDefault(pset);
    }

  private:
    const edm::EDGetTokenT<IntProduct> getToken_;
    edm::EDPutTokenT<IntProduct> putToken_;
    const unsigned int offset_;
    const unsigned int transformOffset_;
  };
}  // namespace edmtest

using edmtest::TransformIntLimitedProducer;
DEFINE_FWK_MODULE(TransformIntLimitedProducer);
