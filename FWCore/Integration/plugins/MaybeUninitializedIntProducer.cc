#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  class MaybeUninitializedIntProducer : public edm::global::EDProducer<> {
  public:
    MaybeUninitializedIntProducer(edm::ParameterSet const& config)
        : value_{config.getParameter<int32_t>("value")}, token_{produces()} {}

    void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const final {
      event.emplace(token_, value_);
    }

  private:
    const cms_int32_t value_;
    const edm::EDPutTokenT<MaybeUninitializedIntProduct> token_;
  };

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::MaybeUninitializedIntProducer);
