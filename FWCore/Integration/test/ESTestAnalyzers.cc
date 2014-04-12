#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Integration/interface/ESTestData.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <algorithm>
#include <vector>

namespace edmtest {

  class ESTestAnalyzerA : public edm::EDAnalyzer {
  public:
    explicit ESTestAnalyzerA(edm::ParameterSet const&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

  private:
    std::vector<int> runsToGetDataFor_;
  };

  ESTestAnalyzerA::ESTestAnalyzerA(edm::ParameterSet const& pset) :
    runsToGetDataFor_(pset.getParameter<std::vector<int> >("runsToGetDataFor")) {
  }

  void ESTestAnalyzerA::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    if (std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run()) != runsToGetDataFor_.end()) {
      ESTestRecordA const& rec = es.get<ESTestRecordA>();
      edm::ESHandle<ESTestDataA> dataA;
      rec.get(dataA);
      edm::LogAbsolute("ESTestAnalyzerA") << "ESTestAnalyzerA: process = " << moduleDescription().processName() << ": Data value = " << dataA->value();
    }
  }



  class ESTestAnalyzerB : public edm::EDAnalyzer {
  public:
    explicit ESTestAnalyzerB(edm::ParameterSet const&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

  private:
    std::vector<int> runsToGetDataFor_;
  };

  ESTestAnalyzerB::ESTestAnalyzerB(edm::ParameterSet const& pset) :
    runsToGetDataFor_(pset.getParameter<std::vector<int> >("runsToGetDataFor")) {
  }

  void ESTestAnalyzerB::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    if (std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run()) != runsToGetDataFor_.end()) {
      ESTestRecordB const& rec = es.get<ESTestRecordB>();
      edm::ESHandle<ESTestDataB> dataB;
      rec.get(dataB);
      edm::LogAbsolute("ESTestAnalyzerB") << "ESTestAnalyzerB: process = " << moduleDescription().processName() << ": Data value = " << dataB->value();
    }
  }

  class ESTestAnalyzerK : public edm::EDAnalyzer {
  public:
    explicit ESTestAnalyzerK(edm::ParameterSet const&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

  private:
    std::vector<int> runsToGetDataFor_;
  };

  ESTestAnalyzerK::ESTestAnalyzerK(edm::ParameterSet const& pset) :
    runsToGetDataFor_(pset.getParameter<std::vector<int> >("runsToGetDataFor")) {
  }

  void ESTestAnalyzerK::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    if (std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run()) != runsToGetDataFor_.end()) {
      ESTestRecordK const& rec = es.get<ESTestRecordK>();
      edm::ESHandle<ESTestDataK> dataK;
      rec.get(dataK);
      edm::LogAbsolute("ESTestAnalyzerK") << "ESTestAnalyzerK: process = " << moduleDescription().processName() << ": Data value = " << dataK->value();
    }
  }

  class ESTestAnalyzerAZ : public edm::EDAnalyzer {
  public:
    explicit ESTestAnalyzerAZ(edm::ParameterSet const&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

  private:
    std::vector<int> runsToGetDataFor_;
  };

  ESTestAnalyzerAZ::ESTestAnalyzerAZ(edm::ParameterSet const& pset) :
    runsToGetDataFor_(pset.getParameter<std::vector<int> >("runsToGetDataFor")) {
  }

  void ESTestAnalyzerAZ::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    if (std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run()) != runsToGetDataFor_.end()) {

      ESTestRecordA const& recA = es.get<ESTestRecordA>();
      edm::ESHandle<ESTestDataA> dataA;
      recA.get("foo", dataA);

      ESTestRecordZ const& recZ = es.get<ESTestRecordZ>();
      edm::ESHandle<ESTestDataZ> dataZ;
      recZ.get("foo", dataZ);

      edm::LogAbsolute("ESTestAnalyzerAZ") << "ESTestAnalyzerAZ: process = " << moduleDescription().processName() << ": Data values = " << dataA->value() << "  " << dataZ->value();
    }
  }
}
using namespace edmtest;
DEFINE_FWK_MODULE(ESTestAnalyzerA);
DEFINE_FWK_MODULE(ESTestAnalyzerB);
DEFINE_FWK_MODULE(ESTestAnalyzerK);
DEFINE_FWK_MODULE(ESTestAnalyzerAZ);
