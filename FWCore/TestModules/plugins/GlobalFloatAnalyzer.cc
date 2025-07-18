#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {

  class GlobalFloatAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit GlobalFloatAnalyzer(edm::ParameterSet const& ps);

    void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDGetTokenT<float> token_;
    float expected_;
  };

  GlobalFloatAnalyzer::GlobalFloatAnalyzer(edm::ParameterSet const& config)
      : token_(consumes(config.getParameter<edm::InputTag>("source"))),
        expected_(static_cast<float>(config.getParameter<double>("expected"))) {}

  void GlobalFloatAnalyzer::analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const& es) const {
    float value = event.get(token_);
    if (value != expected_) {
      throw cms::Exception("LogicError") << "expected value \"" << expected_ << "\"\nreceived value \"" << value << '"';
    }
  }

  void GlobalFloatAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    desc.add<double>("expected", 0.);
    descriptions.addDefault(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::GlobalFloatAnalyzer);
