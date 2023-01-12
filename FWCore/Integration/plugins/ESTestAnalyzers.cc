#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
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
#include "FWCore/Utilities/interface/ESGetToken.h"

#include <algorithm>
#include <vector>

namespace edmtest {

  class ESTestAnalyzerA : public edm::stream::EDAnalyzer<> {
  public:
    explicit ESTestAnalyzerA(edm::ParameterSet const&);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::vector<int> const runsToGetDataFor_;
    std::vector<int> const expectedValues_;
    edm::ESGetToken<ESTestDataA, ESTestRecordA> const token_;
  };

  ESTestAnalyzerA::ESTestAnalyzerA(edm::ParameterSet const& pset)
      : runsToGetDataFor_(pset.getParameter<std::vector<int>>("runsToGetDataFor")),
        expectedValues_(pset.getUntrackedParameter<std::vector<int>>("expectedValues")),
        token_(esConsumes()) {
    assert(expectedValues_.empty() or expectedValues_.size() == runsToGetDataFor_.size());
  }

  void ESTestAnalyzerA::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    auto found = std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run());
    if (found != runsToGetDataFor_.end()) {
      auto const& dataA = es.getData(token_);
      edm::LogAbsolute("ESTestAnalyzerA")
          << "ESTestAnalyzerA: process = " << moduleDescription().processName() << ": Data value = " << dataA.value();
      if (not expectedValues_.empty()) {
        if (expectedValues_[found - runsToGetDataFor_.begin()] != dataA.value()) {
          throw cms::Exception("TestError") << "Exptected value " << expectedValues_[found - runsToGetDataFor_.begin()]
                                            << " but saw " << dataA.value();
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
    desc.addUntracked<std::vector<int>>("expectedValues", std::vector<int>())
        ->setComment("EventSetup value expected for each Run. If empty, no values compared.");
    descriptions.addDefault(desc);
  }

  class ESTestAnalyzerB : public edm::one::EDAnalyzer<> {
  public:
    explicit ESTestAnalyzerB(edm::ParameterSet const&);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    const std::vector<int> runsToGetDataFor_;
    const std::vector<int> expectedValues_;
    const edm::ESGetToken<ESTestDataB, ESTestRecordB> dataBToken_;

    unsigned int expectedIndex_ = 0;
  };

  ESTestAnalyzerB::ESTestAnalyzerB(edm::ParameterSet const& pset)
      : runsToGetDataFor_(pset.getParameter<std::vector<int>>("runsToGetDataFor")),
        expectedValues_(pset.getUntrackedParameter<std::vector<int>>("expectedValues")),
        dataBToken_(esConsumes()) {}

  void ESTestAnalyzerB::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    if (std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run()) != runsToGetDataFor_.end()) {
      edm::ESHandle<ESTestDataB> dataB = es.getHandle(dataBToken_);
      edm::LogAbsolute("ESTestAnalyzerB")
          << "ESTestAnalyzerB: process = " << moduleDescription().processName() << ": Data value = " << dataB->value();

      if (expectedIndex_ < expectedValues_.size()) {
        if (expectedValues_[expectedIndex_] != dataB->value()) {
          throw cms::Exception("TestError")
              << "Expected does not match actual value, "
              << "expected value = " << expectedValues_[expectedIndex_] << " actual value = " << dataB->value();
        }
        ++expectedIndex_;
      }
    }
  }

  void ESTestAnalyzerB::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<int>>("runsToGetDataFor")
        ->setComment("ID number for each Run for which we should get EventSetup data.");
    desc.addUntracked<std::vector<int>>("expectedValues", std::vector<int>())
        ->setComment("EventSetup value expected for each Run. If empty, no values compared.");
    descriptions.addDefault(desc);
  }

  class ESTestAnalyzerK : public edm::global::EDAnalyzer<> {
  public:
    explicit ESTestAnalyzerK(edm::ParameterSet const&);
    void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

  private:
    const std::vector<int> runsToGetDataFor_;
    const edm::ESGetToken<ESTestDataK, ESTestRecordK> dataKToken_;
  };

  ESTestAnalyzerK::ESTestAnalyzerK(edm::ParameterSet const& pset)
      : runsToGetDataFor_(pset.getParameter<std::vector<int>>("runsToGetDataFor")), dataKToken_(esConsumes()) {}

  void ESTestAnalyzerK::analyze(edm::StreamID, edm::Event const& ev, edm::EventSetup const& es) const {
    if (std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run()) != runsToGetDataFor_.end()) {
      ESTestDataK const& dataK = es.getData(dataKToken_);
      edm::LogAbsolute("ESTestAnalyzerK")
          << "ESTestAnalyzerK: process = " << moduleDescription().processName() << ": Data value = " << dataK.value();
    }
  }

  class ESTestAnalyzerAZ : public edm::global::EDAnalyzer<> {
  public:
    explicit ESTestAnalyzerAZ(edm::ParameterSet const&);
    void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::vector<int> const runsToGetDataFor_;
    std::vector<int> const expectedValuesA_;
    std::vector<int> const expectedValuesZ_;
    edm::ESGetToken<ESTestDataA, ESTestRecordA> dataAToken_;
    edm::ESGetToken<ESTestDataZ, ESTestRecordZ> dataZToken_;
  };

  ESTestAnalyzerAZ::ESTestAnalyzerAZ(edm::ParameterSet const& pset)
      : runsToGetDataFor_(pset.getParameter<std::vector<int>>("runsToGetDataFor")),
        expectedValuesA_(pset.getUntrackedParameter<std::vector<int>>("expectedValuesA")),
        expectedValuesZ_(pset.getUntrackedParameter<std::vector<int>>("expectedValuesZ")),
        dataAToken_(esConsumes(edm::ESInputTag("", "foo"))),
        dataZToken_(esConsumes(edm::ESInputTag("", "foo"))) {
    assert(expectedValuesA_.empty() or expectedValuesA_.size() == runsToGetDataFor_.size());
    assert(expectedValuesZ_.empty() or expectedValuesZ_.size() == runsToGetDataFor_.size());
  }

  void ESTestAnalyzerAZ::analyze(edm::StreamID, edm::Event const& ev, edm::EventSetup const& es) const {
    auto found = std::find(runsToGetDataFor_.begin(), runsToGetDataFor_.end(), ev.run());

    if (found != runsToGetDataFor_.end()) {
      ESTestDataA const& dataA = es.getData(dataAToken_);

      ESTestDataZ const& dataZ = es.getData(dataZToken_);

      edm::LogAbsolute("ESTestAnalyzerAZ") << "ESTestAnalyzerAZ: process = " << moduleDescription().processName()
                                           << ": Data values = " << dataA.value() << "  " << dataZ.value();

      if (not expectedValuesA_.empty()) {
        if (expectedValuesA_[found - runsToGetDataFor_.begin()] != dataA.value()) {
          throw cms::Exception("TestError")
              << "Exptected value for A " << expectedValuesA_[found - runsToGetDataFor_.begin()] << " but saw "
              << dataA.value();
        }
      }

      if (not expectedValuesZ_.empty()) {
        if (expectedValuesZ_[found - runsToGetDataFor_.begin()] != dataZ.value()) {
          throw cms::Exception("TestError")
              << "Exptected value for Z " << expectedValuesZ_[found - runsToGetDataFor_.begin()] << " but saw "
              << dataZ.value();
        }
      }
    }
  }

  void ESTestAnalyzerAZ::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Test module for the EventSetup");
    desc.add<std::vector<int>>("runsToGetDataFor")
        ->setComment("ID number for each Run for which we should get EventSetup data.");
    desc.addUntracked<std::vector<int>>("expectedValuesA", std::vector<int>())
        ->setComment("EventSetup value for ESTestDataA:foo expected for each Run. If empty, no values compared.");
    desc.addUntracked<std::vector<int>>("expectedValuesZ", std::vector<int>())
        ->setComment("EventSetup value for ESTestDataZ:foo expected for each Run. If empty, no values compared.");
    descriptions.addDefault(desc);
  }

  class ESTestAnalyzerJ : public edm::stream::EDAnalyzer<> {
  public:
    explicit ESTestAnalyzerJ(edm::ParameterSet const&) : dataJToken_(esConsumes()) {}
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    const edm::ESGetToken<ESTestDataJ, ESTestRecordJ> dataJToken_;
  };

  void ESTestAnalyzerJ::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    ESTestDataJ const& dataJ = es.getData(dataJToken_);
    edm::LogAbsolute("ESTestAnalyzerJ") << "ESTestAnalyzerJ: process = " << moduleDescription().processName()
                                        << ": Data values = " << dataJ.value();
  }

  class ESTestAnalyzerL : public edm::stream::EDAnalyzer<> {
  public:
    explicit ESTestAnalyzerL(edm::ParameterSet const& iConfig)
        : edToken_(consumes(iConfig.getParameter<edm::InputTag>("src"))), esToken_(esConsumes()) {}
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    edm::EDGetTokenT<IntProduct> edToken_;
    edm::ESGetToken<ESTestDataJ, ESTestRecordJ> esToken_;
  };

  void ESTestAnalyzerL::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    auto const& intData = ev.get(edToken_);
    auto const& dataJ = es.getData(esToken_);
    edm::LogAbsolute("ESTestAnalyzerJ") << "ESTestAnalyzerL: process = " << moduleDescription().processName()
                                        << ": ED value " << intData.value << ": ES value = " << dataJ.value();
  }

  class ESTestAnalyzerIncorrectConsumes : public edm::stream::EDAnalyzer<> {
  public:
    explicit ESTestAnalyzerIncorrectConsumes(edm::ParameterSet const& iConfig){};
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    edm::ESGetToken<ESTestDataJ, ESTestRecordJ> esToken_;
  };

  void ESTestAnalyzerIncorrectConsumes::analyze(edm::Event const& ev, edm::EventSetup const& es) {
    esToken_ = esConsumes();
    edm::LogAbsolute("ESTestAnalyzerIncorrectConsumes")
        << "Succeeded to call esConsumes() in analyze(), should not happen!";
  }

}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_MODULE(ESTestAnalyzerA);
DEFINE_FWK_MODULE(ESTestAnalyzerB);
DEFINE_FWK_MODULE(ESTestAnalyzerK);
DEFINE_FWK_MODULE(ESTestAnalyzerAZ);
DEFINE_FWK_MODULE(ESTestAnalyzerJ);
DEFINE_FWK_MODULE(ESTestAnalyzerL);
DEFINE_FWK_MODULE(ESTestAnalyzerIncorrectConsumes);
