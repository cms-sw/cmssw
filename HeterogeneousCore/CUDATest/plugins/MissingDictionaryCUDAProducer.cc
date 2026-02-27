#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDATest/interface/MissingDictionaryCUDAObject.h"

namespace edmtest {

  class MissingDictionaryCUDAProducer : public edm::global::EDProducer<> {
  public:
    explicit MissingDictionaryCUDAProducer(edm::ParameterSet const& config) : token_(produces()) {}

    void produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& setup) const final {
      event.emplace(token_);
    }

  private:
    const edm::EDPutTokenT<MissingDictionaryCUDAObject> token_;
  };

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::MissingDictionaryCUDAProducer);
