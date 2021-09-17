#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <sstream>

namespace edmtest {

  class UTC_S1 : public edm::one::EDAnalyzer<> {
  public:
    explicit UTC_S1(edm::ParameterSet const &pset) {
      identifier = pset.getUntrackedParameter<int>("identifier", 99);
      edm::GroupLogStatistics("grouped_cat");
    }

    void analyze(edm::Event const &, edm::EventSetup const &) override;

  private:
    int identifier;
    bool enableNotYetCalled = true;
    int n = 0;
  };

  class UTC_S2 : public edm::one::EDAnalyzer<> {
  public:
    explicit UTC_S2(edm::ParameterSet const &p) { identifier = p.getUntrackedParameter<int>("identifier", 98); }

    void analyze(edm::Event const &, edm::EventSetup const &) override;

  private:
    int identifier;
    int n = 0;
  };

  class UTC_SUMMARY : public edm::global::EDAnalyzer<> {
  public:
    explicit UTC_SUMMARY(edm::ParameterSet const &) {}

    void analyze(edm::StreamID, edm::Event const &, edm::EventSetup const &) const override;
  };

  void UTC_S1::analyze(edm::Event const &, edm::EventSetup const &) {
    if (enableNotYetCalled) {
      edm::EnableLoggedErrorsSummary();
      enableNotYetCalled = false;
    }
    n++;
    if (n <= 2)
      return;
    edm::LogError("cat_A") << "S1 with identifier " << identifier << " n = " << n;
    edm::LogError("grouped_cat") << "S1 timer with identifier " << identifier;
  }

  void UTC_S2::analyze(edm::Event const &, edm::EventSetup const &) {
    n++;
    if (n <= 2)
      return;
    edm::LogError("cat_A") << "S2 with identifier " << identifier;
    edm::LogError("grouped_cat") << "S2 timer with identifier " << identifier;
    edm::LogError("cat_B") << "S2B with identifier " << identifier;
    for (int i = 0; i < n; ++i) {
      edm::LogError("cat_B") << "more S2B";
    }
  }

  void UTC_SUMMARY::analyze(edm::StreamID, edm::Event const &iEvent, edm::EventSetup const &) const {
    const auto index = iEvent.streamID().value();
    if (!edm::FreshErrorsExist(index)) {
      edm::LogInfo("NoFreshErrors") << "Not in this event, anyway";
    }
    auto es = edm::LoggedErrorsSummary(index);
    std::ostringstream os;
    for (unsigned int i = 0; i != es.size(); ++i) {
      os << es[i].category << "   " << es[i].module << "   " << es[i].count << "\n";
    }
    edm::LogVerbatim("ErrorsInEvent") << os.str();
  }

}  // namespace edmtest

using edmtest::UTC_S1;
using edmtest::UTC_S2;
using edmtest::UTC_SUMMARY;
DEFINE_FWK_MODULE(UTC_S1);
DEFINE_FWK_MODULE(UTC_S2);
DEFINE_FWK_MODULE(UTC_SUMMARY);
