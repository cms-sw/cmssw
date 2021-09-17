#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <sstream>
#include <vector>

namespace edmtest {

  class UTC_T1 : public edm::one::EDAnalyzer<> {
  public:
    explicit UTC_T1(edm::ParameterSet const& ps) : ev(0) {
      identifier = ps.getUntrackedParameter<int>("identifier", 99);
    }

    void analyze(edm::Event const&, edm::EventSetup const&) override;

  private:
    int identifier;
    int ev;
  };

  class UTC_T2 : public edm::one::EDAnalyzer<> {
  public:
    explicit UTC_T2(edm::ParameterSet const& ps) : ev(0) {
      identifier = ps.getUntrackedParameter<int>("identifier", 98);
    }

    void analyze(edm::Event const&, edm::EventSetup const&) override;

  private:
    int identifier;
    int ev;
    void printLES(std::vector<edm::messagelogger::ErrorSummaryEntry> const& v);
  };

  void UTC_T1::analyze(edm::Event const&, edm::EventSetup const&) {
    if (ev == 0)
      edm::EnableLoggedErrorsSummary();
    edm::LogError("cat_A") << "T1 error with identifier " << identifier << " event " << ev;
    edm::LogWarning("cat_A") << "T1 warning with identifier " << identifier << " event " << ev;
    edm::LogError("timer") << "T1 timer error with identifier " << identifier << " event " << ev;
    ev++;
  }

  void UTC_T2::analyze(edm::Event const& iEvent, edm::EventSetup const& /*unused*/
  ) {
    const auto index = iEvent.streamID().value();
    edm::LogError("cat_A") << "T2 error with identifier " << identifier << " event " << ev;
    edm::LogWarning("cat_A") << "T2 warning with identifier " << identifier << " event " << ev;
    edm::LogError("timer") << "T2 timer error with identifier " << identifier << " event " << ev;
    if (ev == 9) {
      if (edm::FreshErrorsExist(index)) {
        edm::LogInfo("summary") << "At ev = " << ev << "FreshErrorsExist() returns true";
      } else {
        edm::LogError("summary") << "At ev = " << ev << "FreshErrorsExist() returns false"
                                 << " which is unexpected";
      }
      auto v = edm::LoggedErrorsSummary(index);
      printLES(v);
    }
    if (ev == 15) {
      if (edm::FreshErrorsExist(index)) {
        edm::LogInfo("summary") << "At ev = " << ev << "FreshErrorsExist() returns true";
      } else {
        edm::LogError("summary") << "At ev = " << ev << "FreshErrorsExist() returns false"
                                 << " which is unexpected";
      }
      auto v = edm::LoggedErrorsOnlySummary(index);
      printLES(v);
    }
    ev++;
  }

  void UTC_T2::printLES(std::vector<edm::messagelogger::ErrorSummaryEntry> const& v) {
    std::ostringstream s;
    auto end = v.end();
    s << "Error Summary Vector with " << v.size() << " entries:\n";
    for (auto i = v.begin(); i != end; ++i) {
      s << "Category " << i->category << "   Module " << i->module << "   Severity " << (i->severity).getName()
        << "   Count " << i->count << "\n";
    }
    s << "-------------------------- \n";
    edm::LogVerbatim("summary") << s.str();
  }

}  // namespace edmtest

using edmtest::UTC_T1;
using edmtest::UTC_T2;
DEFINE_FWK_MODULE(UTC_T1);
DEFINE_FWK_MODULE(UTC_T2);
