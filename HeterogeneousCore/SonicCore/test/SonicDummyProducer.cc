#include "DummyClient.h"
#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <memory>

namespace sonictest {
  class SonicDummyProducer : public SonicEDProducer<DummyClient> {
  public:
    explicit SonicDummyProducer(edm::ParameterSet const& cfg)
        : SonicEDProducer<DummyClient>(cfg), input_(cfg.getParameter<int>("input")) {
      putToken_ = produces<edmtest::IntProduct>();
    }

    void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override { iInput = input_; }

    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
      iEvent.emplace(putToken_, iOutput);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      DummyClient::fillPSetDescription(desc);
      desc.add<int>("input");
      //to ensure distinct cfi names
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    //members
    int input_;
    edm::EDPutTokenT<edmtest::IntProduct> putToken_;
  };
}  // namespace sonictest

using sonictest::SonicDummyProducer;
DEFINE_FWK_MODULE(SonicDummyProducer);
