#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {

  class GlobalStringAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit GlobalStringAnalyzer(edm::ParameterSet const& ps);

    void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDGetTokenT<std::string> token_;
    std::string expected_;
  };

  GlobalStringAnalyzer::GlobalStringAnalyzer(edm::ParameterSet const& config)
      : token_(consumes(config.getParameter<edm::InputTag>("source"))),
        expected_(config.getParameter<std::string>("expected")) {}

  void GlobalStringAnalyzer::analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const& es) const {
    std::string const& value = event.get(token_);
    if (value != expected_) {
      std::string msg = "GlobalStringAnalyzer: error\nexpected value \"";
      msg += expected_;
      msg += "\"\nreceived value \"";
      msg += value;
      msg += "\"";
      throw msg;
    }
  }

  void GlobalStringAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    desc.add<std::string>("expected", "Hello world");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::GlobalStringAnalyzer);
