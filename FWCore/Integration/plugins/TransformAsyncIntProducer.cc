#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <thread>
namespace edmtest {
  class TransformAsyncIntProducer : public edm::global::EDProducer<edm::Transformer> {
  public:
    struct WorkCache {
      std::shared_ptr<std::thread> thread_;
      IntProduct value_;
    };

    TransformAsyncIntProducer(edm::ParameterSet const& iPSet)
        : getToken_(consumes(iPSet.getParameter<edm::InputTag>("get"))),
          offset_(iPSet.getParameter<unsigned int>("offset")),
          transformOffset_(iPSet.getParameter<unsigned int>("transformOffset")),
          noPut_(iPSet.getParameter<bool>("noPut")) {
      putToken_ = produces<IntProduct>();
      bool check = iPSet.getUntrackedParameter<bool>("checkTransformNotCalled");
      registerTransformAsync(
          putToken_,
          [offset = transformOffset_, check](auto const& iFrom, auto iTask) {
            if (check) {
              throw cms::Exception("TransformShouldNotBeCalled");
            }
            WorkCache ret;
            ret.thread_ = std::make_shared<std::thread>([iTask] { usleep(100000); });
            ret.value_ = IntProduct(iFrom.value + offset);
            return ret;
          },
          [](auto const& iFrom) {
            iFrom.thread_->join();
            return iFrom.value_;
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

using edmtest::TransformAsyncIntProducer;
DEFINE_FWK_MODULE(TransformAsyncIntProducer);
