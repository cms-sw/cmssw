#ifndef EDM_ML_DEBUG
#define EDM_ML_DEBUG
#endif

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {

  class UTC_Vd1 : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
  public:
    explicit UTC_Vd1(edm::ParameterSet const& ps) : ev(0) {
      identifier = ps.getUntrackedParameter<int>("identifier", 99);
    }

    void analyze(edm::Event const&, edm::EventSetup const&) override;

    void beginJob() override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void endRun(edm::Run const&, edm::EventSetup const&) override;
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  private:
    int identifier;
    int ev;
  };

  class UTC_Vd2 : public edm::one::EDAnalyzer<> {
  public:
    explicit UTC_Vd2(edm::ParameterSet const& ps) : ev(0) {
      identifier = ps.getUntrackedParameter<int>("identifier", 98);
    }

    void analyze(edm::Event const&, edm::EventSetup const&) override;

  private:
    int identifier;
    int ev;
  };

  void UTC_Vd1::analyze(edm::Event const&, edm::EventSetup const&) {
    edm::LogError("cat_A") << "T1 analyze error with identifier " << identifier << " event " << ev;
    edm::LogWarning("cat_A") << "T1 analyze warning with identifier " << identifier << " event " << ev;
    edm::LogInfo("cat_A") << "T1 analyze info with identifier " << identifier << " event " << ev;
    LogDebug("cat_A") << "T1 analyze debug with identifier " << identifier << " event " << ev;
    ev++;
  }

  void UTC_Vd1::beginJob() {
    edm::LogWarning("cat_BJ") << "T1 beginJob warning with identifier " << identifier << " event " << ev;
    LogDebug("cat_BJ") << "T1 beginJob debug with identifier " << identifier << " event " << ev;
  }

  void UTC_Vd1::beginRun(edm::Run const&, edm::EventSetup const&) {
    edm::LogInfo("cat_BR") << "T1 beginRun info with identifier " << identifier << " event " << ev;
    LogDebug("cat_BJ") << "T1 beginRun debug with identifier " << identifier << " event " << ev;
  }

  void UTC_Vd1::endRun(edm::Run const&, edm::EventSetup const&) {}

  void UTC_Vd1::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
    edm::LogWarning("cat_BL") << "T1 beginLumi warning with identifier " << identifier << " event " << ev;
    LogDebug("cat_BL") << "T1 beginLumi debug with identifier " << identifier << " event " << ev;
  }

  void UTC_Vd1::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

  void UTC_Vd2::analyze(edm::Event const&, edm::EventSetup const&) {
    edm::LogError("cat_A") << "T1 analyze error with identifier " << identifier << " event " << ev;
    edm::LogWarning("cat_A") << "T1 analyze warning with identifier " << identifier << " event " << ev;
    edm::LogInfo("cat_A") << "T1 analyze info with identifier " << identifier << " event " << ev;
    ev++;
  }

}  // end namespace edmtest

using edmtest::UTC_Vd1;
using edmtest::UTC_Vd2;
DEFINE_FWK_MODULE(UTC_Vd1);
DEFINE_FWK_MODULE(UTC_Vd2);
