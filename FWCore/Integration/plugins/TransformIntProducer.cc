#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edmtest {
  class TransformIntProducer : public edm::global::EDProducer<edm::Transformer> {
  public:
    TransformIntProducer(edm::ParameterSet const& iPSet)
        : getToken_(consumes(iPSet.getParameter<edm::InputTag>("get"))),
          offset_(iPSet.getParameter<unsigned int>("offset")),
          transformOffset_(iPSet.getParameter<unsigned int>("transformOffset")),
          noPut_(iPSet.getParameter<bool>("noPut")) {
      putToken_ = produces<IntProduct>();
      bool check = iPSet.getUntrackedParameter<bool>("checkTransformNotCalled");
      registerTransform(
          putToken_,
          [offset = transformOffset_, check](auto const& iFrom) {
            if (check) {
              throw cms::Exception("TransformShouldNotBeCalled");
            }
            return IntProduct(iFrom.value + offset);
          },
          "transform");
    }

    void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
      if (not noPut_) {
        iEvent.emplace(putToken_, iEvent.get(getToken_).value + offset_);
      }
    }
    static void fillDescriptions(edm::ConfigurationDescriptions& desc) {
      edm::ParameterSetDescription pset;
      pset.add<edm::InputTag>("get");
      pset.add<unsigned int>("offset", 0);
      pset.add<unsigned int>("transformOffset", 1);
      pset.addUntracked<bool>("checkTransformNotCalled", false);
      pset.add<bool>("noPut", false);

      desc.addDefault(pset);
    }

  private:
    const edm::EDGetTokenT<IntProduct> getToken_;
    edm::EDPutTokenT<IntProduct> putToken_;
    const unsigned int offset_;
    const unsigned int transformOffset_;
    const bool noPut_;
  };
}  // namespace edmtest

using edmtest::TransformIntProducer;
DEFINE_FWK_MODULE(TransformIntProducer);
