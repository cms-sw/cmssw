#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <string>

namespace edmtest {

  class UTC_Q1 : public edm::global::EDAnalyzer<> {
  public:
    explicit UTC_Q1(edm::ParameterSet const& p) {
      identifier = p.getUntrackedParameter<int>("identifier", 99);
      edm::GroupLogStatistics("timer");  // these lines would normally be in
      edm::GroupLogStatistics("trace");  // whatever service knows that
                                         // timer and trace should be groupd
                                         // by moduels for statistics
    }

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

  private:
    int identifier;
  };

  void UTC_Q1::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    edm::LogInfo("cat_A") << "Q1 with identifier " << identifier;
    edm::LogInfo("timer") << "Q1 timer with identifier " << identifier;
    edm::LogInfo("trace") << "Q1 trace with identifier " << identifier;
  }

  class UTC_Q2 : public edm::global::EDAnalyzer<> {
  public:
    explicit UTC_Q2(edm::ParameterSet const& p) { identifier = p.getUntrackedParameter<int>("identifier", 98); }

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

  private:
    int identifier;
  };

  void UTC_Q2::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    edm::LogInfo("cat_A") << "Q2 with identifier " << identifier;
    edm::LogInfo("timer") << "Q2 timer with identifier " << identifier;
    edm::LogInfo("trace") << "Q2 trace with identifier " << identifier;
  }

}  // namespace edmtest

using edmtest::UTC_Q1;
using edmtest::UTC_Q2;
DEFINE_FWK_MODULE(UTC_Q1);
DEFINE_FWK_MODULE(UTC_Q2);
