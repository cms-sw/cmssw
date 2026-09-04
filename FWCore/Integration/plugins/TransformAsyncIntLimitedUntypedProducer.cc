#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/limited/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <chrono>
#include <memory>
#include <thread>

namespace edmtest {
  class TransformAsyncIntLimitedUntypedProducer : public edm::limited::EDProducer<edm::Transformer> {
  public:
    struct WorkCache {
      std::shared_ptr<std::thread> thread_;
      IntProduct value_;
    };

    TransformAsyncIntLimitedUntypedProducer(edm::ParameterSet const& iPSet)
        : edm::limited::EDProducerBase(iPSet),
          edm::limited::EDProducer<edm::Transformer>(iPSet),
          getToken_(consumes(iPSet.getParameter<edm::InputTag>("get"))),
          offset_(iPSet.getParameter<unsigned int>("offset")),
          transformOffset_(iPSet.getParameter<unsigned int>("transformOffset")),
          noPut_(iPSet.getParameter<bool>("noPut")) {
      putToken_ = produces<IntProduct>();
      bool check = iPSet.getUntrackedParameter<bool>("checkTransformNotCalled");
      registerTransformAsync(
          putToken_,
          [offset = transformOffset_, check](edm::StreamID, edm::WrapperBase const& iGotProduct, auto iTask) {
            if (check) {
              throw cms::Exception("TransformShouldNotBeCalled");
            }
            auto const* product = static_cast<edm::Wrapper<IntProduct> const&>(iGotProduct).product();
            WorkCache ret;
            using namespace std::chrono_literals;
            ret.thread_ = std::make_shared<std::thread>([iTask] { std::this_thread::sleep_for(100ms); });
            ret.value_ = IntProduct(product->value + offset);
            return ret;
          },
          [](edm::StreamID, WorkCache work) -> std::unique_ptr<edm::WrapperBase> {
            work.thread_->join();
            return std::make_unique<edm::Wrapper<IntProduct>>(edm::WrapperBase::Emplace{}, work.value_);
          },
          edm::TypeID(typeid(IntProduct)),
          "transform");
    }

    void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
      if (not noPut_) {
        std::unique_ptr<edm::WrapperBase> product = std::make_unique<edm::Wrapper<IntProduct>>(
            edm::WrapperBase::Emplace{}, iEvent.get(getToken_).value + offset_);
        iEvent.put(putToken_, std::move(product));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& desc) {
      edm::ParameterSetDescription pset;
      pset.add<edm::InputTag>("get");
      pset.add<unsigned int>("offset", 0);
      pset.add<unsigned int>("transformOffset", 1);
      pset.addUntracked<bool>("checkTransformNotCalled", false);
      pset.addUntracked<unsigned int>("concurrencyLimit", 1);
      pset.add<bool>("noPut", false);

      desc.addDefault(pset);
    }

  private:
    const edm::EDGetTokenT<IntProduct> getToken_;
    edm::EDPutToken putToken_;
    const unsigned int offset_;
    const unsigned int transformOffset_;
    const bool noPut_;
  };
}  // namespace edmtest

using edmtest::TransformAsyncIntLimitedUntypedProducer;
DEFINE_FWK_MODULE(TransformAsyncIntLimitedUntypedProducer);
