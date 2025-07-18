#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace {

  template <typename T>
  T& operator<<(T& out, std::vector<double> const& values) {
    if (values.empty()) {
      out << "{}";
      return out;
    }

    auto it = values.begin();
    out << "{ " << *it;
    ++it;
    while (it != values.end()) {
      out << ", " << *it;
      ++it;
    }
    out << " }";
    return out;
  }

}  // namespace

namespace edmtest {

  class GlobalVectorAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit GlobalVectorAnalyzer(edm::ParameterSet const& ps);

    void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDGetTokenT<std::vector<double>> token_;
    std::vector<double> expected_;
  };

  GlobalVectorAnalyzer::GlobalVectorAnalyzer(edm::ParameterSet const& config)
      : token_(consumes(config.getParameter<edm::InputTag>("source"))),
        expected_(config.getParameter<std::vector<double>>("expected")) {}

  void GlobalVectorAnalyzer::analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const& es) const {
    std::vector<double> const& values = event.get(token_);
    if (values != expected_) {
      throw cms::Exception("LogicError") << "expected values \"" << expected_ << "\"\nreceived values \"" << values
                                         << '"';
    }
  }

  void GlobalVectorAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    desc.add<std::vector<double>>("expected", {});
    descriptions.addDefault(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::GlobalVectorAnalyzer);
