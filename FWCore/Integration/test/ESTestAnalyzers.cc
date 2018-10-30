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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <algorithm>
#include <vector>

namespace edmtest {

  class ESTestAnalyzerA : public edm::EDAnalyzer {
  public:
    explicit ESTestAnalyzerA(edm::ParameterSet const&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  private:
    std::vector<int> runsToGetDataFor_;
    std::vector<int> expectedValues_;
  };

  ESTestAnalyzerA::ESTestAnalyzerA(edm::ParameterSet const& pset) :
    runsToGetDataFor_(pset.getParameter<std::vector<int> >("runsToGetDataFor")),
    expectedValues_(pset.getUntrackedParameter<std::vector<int>>("expectedValues")){
      assert( expectedValues_.empty() or expectedValues_.size() == runsToGetDataFor_.size());
  }

  void ESTestAnalyzerA::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    auto found = std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run());
    if ( found != runsToGetDataFor_.end()) {
      ESTestRecordA const& rec = es.get<ESTestRecordA>();
      edm::ESHandle<ESTestDataA> dataA;
      rec.get(dataA);
      edm::LogAbsolute("ESTestAnalyzerA") << "ESTestAnalyzerA: process = " << moduleDescription().processName() << ": Data value = " << dataA->value();
      if(not expectedValues_.empty()) {
        if(expectedValues_[found-runsToGetDataFor_.begin()] != dataA->value()) {
          throw cms::Exception("TestError")<<"Exptected value "<<expectedValues_[found-runsToGetDataFor_.begin()]<<" but saw "<<dataA->value();
        }
      }
    }
  }

  void ESTestAnalyzerA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    desc.setComment("Test module for the EventSetup");
    desc.add<std::vector<int>>("runsToGetDataFor")
    ->setComment("ID number for each Run for which we should get EventSetup data.");
    desc.addUntracked<std::vector<int>>("expectedValues",std::vector<int>())
    ->setComment("EventSetup value expected for each Run. If empty, no values compared.");
    descriptions.addDefault(desc);
  }



  class ESTestAnalyzerB : public edm::EDAnalyzer {
  public:
    explicit ESTestAnalyzerB(edm::ParameterSet const&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::vector<int> runsToGetDataFor_;
    std::vector<int> expectedValues_;

    unsigned int expectedIndex_ = 0;
  };

  ESTestAnalyzerB::ESTestAnalyzerB(edm::ParameterSet const& pset) :
    runsToGetDataFor_(pset.getParameter<std::vector<int> >("runsToGetDataFor")),
    expectedValues_(pset.getUntrackedParameter<std::vector<int> >("expectedValues")) {
  }

  void ESTestAnalyzerB::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    if (std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run()) != runsToGetDataFor_.end()) {
      ESTestRecordB const& rec = es.get<ESTestRecordB>();
      edm::ESHandle<ESTestDataB> dataB;
      rec.get(dataB);
      edm::LogAbsolute("ESTestAnalyzerB") << "ESTestAnalyzerB: process = " << moduleDescription().processName() << ": Data value = " << dataB->value();

      if (expectedIndex_ < expectedValues_.size()) {
        if (expectedValues_[expectedIndex_] != dataB->value()) {
          throw cms::Exception("TestError") << "Expected does not match actual value, "
            << "expected value = " << expectedValues_[expectedIndex_]
            << " actual value = " << dataB->value();
        }
        ++expectedIndex_;
      }
    }
  }

  void ESTestAnalyzerB::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<int>>("runsToGetDataFor")
    ->setComment("ID number for each Run for which we should get EventSetup data.");
    desc.addUntracked<std::vector<int>>("expectedValues",std::vector<int>())
    ->setComment("EventSetup value expected for each Run. If empty, no values compared.");
    descriptions.addDefault(desc);
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
