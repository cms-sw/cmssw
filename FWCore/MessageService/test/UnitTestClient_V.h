#ifndef FWCore_MessageService_test_UnitTestClient_V_h
#define FWCore_MessageService_test_UnitTestClient_V_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

namespace edm {
  class ParameterSet;
}

namespace edmtest {

  class UTC_V1
      : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks, edm::WatchProcessBlock> {
  public:
    explicit UTC_V1(edm::ParameterSet const& p) : ev(0) { identifier = p.getUntrackedParameter<int>("identifier", 99); }

    ~UTC_V1() override {}

    void analyze(edm::Event const& e, edm::EventSetup const& c) override;

    void beginJob() override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}
    void endRun(edm::Run const&, edm::EventSetup const&) override {}

    void beginProcessBlock(edm::ProcessBlock const&) override;
    void endProcessBlock(edm::ProcessBlock const&) override;

  private:
    int identifier;
    int ev;
  };

  class UTC_V2 : public edm::EDAnalyzer {
  public:
    explicit UTC_V2(edm::ParameterSet const& p) : ev(0) { identifier = p.getUntrackedParameter<int>("identifier", 98); }

    ~UTC_V2() override {}

    void analyze(edm::Event const& e, edm::EventSetup const& c) override;

  private:
    int identifier;
    int ev;
  };

}  // namespace edmtest

#endif  // FWCore_MessageService_test_UnitTestClient_T_h
