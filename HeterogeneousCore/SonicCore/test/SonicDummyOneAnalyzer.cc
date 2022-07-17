#include "DummyClient.h"
#include "HeterogeneousCore/SonicCore/interface/SonicOneEDAnalyzer.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>

namespace sonictest {
  //designed similar to IntTestAnalyzer
  class SonicDummyOneAnalyzer : public SonicOneEDAnalyzer<DummyClient> {
  public:
    explicit SonicDummyOneAnalyzer(edm::ParameterSet const& cfg)
        : SonicOneEDAnalyzer<DummyClient>(cfg),
          input_(cfg.getParameter<int>("input")),
          expected_(cfg.getParameter<int>("expected")) {}

    void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override { iInput = input_; }

    void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
      if (iOutput != expected_)
        throw cms::Exception("ValueMismatch")
            << "The value is " << iOutput << " but it was supposed to be " << expected_;
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      DummyClient::fillPSetDescription(desc);
      desc.add<int>("input");
      desc.add<int>("expected");
      //to ensure distinct cfi names
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    //members
    int input_, expected_;
  };
}  // namespace sonictest

using sonictest::SonicDummyOneAnalyzer;
DEFINE_FWK_MODULE(SonicDummyOneAnalyzer);
