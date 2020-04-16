#include "DummyClient.h"
#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <memory>

namespace sonictest {
  template <typename Client>
  class SonicDummyProducer : public SonicEDProducer<Client> {
  public:
    //needed because base class has dependent scope
    using typename SonicEDProducer<Client>::Input;
    using typename SonicEDProducer<Client>::Output;
    explicit SonicDummyProducer(edm::ParameterSet const& cfg)
        : SonicEDProducer<Client>(cfg), input_(cfg.getParameter<int>("input")) {
      //for debugging
      this->setDebugName("SonicDummyProducer");
      putToken_ = this->template produces<edmtest::IntProduct>();
    }

    void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override { iInput = input_; }

    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
      iEvent.emplace(putToken_, iOutput);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      Client::fillPSetDescription(desc);
      desc.add<int>("input");
      //to ensure distinct cfi names
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    //members
    int input_;
    edm::EDPutTokenT<edmtest::IntProduct> putToken_;
  };

  typedef SonicDummyProducer<DummyClientSync> SonicDummyProducerSync;
  typedef SonicDummyProducer<DummyClientPseudoAsync> SonicDummyProducerPseudoAsync;
  typedef SonicDummyProducer<DummyClientAsync> SonicDummyProducerAsync;
}  // namespace sonictest

using sonictest::SonicDummyProducerSync;
DEFINE_FWK_MODULE(SonicDummyProducerSync);
using sonictest::SonicDummyProducerPseudoAsync;
DEFINE_FWK_MODULE(SonicDummyProducerPseudoAsync);
using sonictest::SonicDummyProducerAsync;
DEFINE_FWK_MODULE(SonicDummyProducerAsync);
