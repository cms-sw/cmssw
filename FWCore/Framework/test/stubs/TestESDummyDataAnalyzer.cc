// -*- C++ -*-
//
// Package:    Framework
// Class:      TestESDummyDataAnalyzer
//
/**\class TestESDummyDataAnalyzer TestESDummyDataAnalyzer.cc FWCore/Framework/test/stubs/TestESDummyDataAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Dec 22 11:02:00 EST 2005
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/test/DummyData.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace testesdummydata {
  struct DummyRunCache {};
}  // namespace testesdummydata

using testesdummydata::DummyRunCache;

class TestESDummyDataAnalyzer : public edm::one::EDAnalyzer<edm::RunCache<DummyRunCache>> {
public:
  explicit TestESDummyDataAnalyzer(const edm::ParameterSet&);

private:
  void endJob() override;
  std::shared_ptr<DummyRunCache> globalBeginRun(const edm::Run&, const edm::EventSetup&) const override { return {}; }
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void globalEndRun(const edm::Run&, const edm::EventSetup&) override {}

  int m_expectedValue;
  int const m_nEventsValue;
  int m_counter{};
  int m_totalCounter{};
  int const m_totalNEvents;
  // At the moment the begin run token is not used to get anything,
  // but just its existence tests the indexing used in the esConsumes
  // function call
  edm::ESGetToken<edm::eventsetup::test::DummyData, edm::DefaultRecord> const m_esTokenBeginRun;
  edm::ESGetToken<edm::eventsetup::test::DummyData, edm::DefaultRecord> const m_esToken;
};

TestESDummyDataAnalyzer::TestESDummyDataAnalyzer(const edm::ParameterSet& iConfig)
    : m_expectedValue{iConfig.getParameter<int>("expected")},
      m_nEventsValue{iConfig.getUntrackedParameter<int>("nEvents", 0)},
      m_totalNEvents{iConfig.getUntrackedParameter<int>("totalNEvents", -1)},
      m_esTokenBeginRun{esConsumes<edm::Transition::BeginRun>()},
      m_esToken{esConsumes()} {}

void TestESDummyDataAnalyzer::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  using namespace edm;

  ++m_totalCounter;
  if (m_nEventsValue) {
    ++m_counter;
    if (m_nEventsValue < m_counter) {
      ++m_expectedValue;
      m_counter = 0;
    }
  }

  ESHandle<edm::eventsetup::test::DummyData> pData = iSetup.getHandle(m_esToken);

  if (m_expectedValue != pData->value_) {
    throw cms::Exception("WrongValue") << "got value " << pData->value_ << " but expected " << m_expectedValue;
  }
}

void TestESDummyDataAnalyzer::endJob() {
  if (-1 != m_totalNEvents && m_totalNEvents != m_totalCounter) {
    throw cms::Exception("WrongNumberOfEvents")
        << "expected " << m_totalNEvents << " but instead saw " << m_totalCounter << "\n";
  }
}

class TestESDummyDataNoPrefetchAnalyzer : public edm::one::EDAnalyzer<edm::RunCache<DummyRunCache>> {
public:
  explicit TestESDummyDataNoPrefetchAnalyzer(const edm::ParameterSet&) {}

private:
  std::shared_ptr<DummyRunCache> globalBeginRun(const edm::Run&, const edm::EventSetup&) const override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void globalEndRun(const edm::Run&, const edm::EventSetup&) override {}
};

std::shared_ptr<DummyRunCache> TestESDummyDataNoPrefetchAnalyzer::globalBeginRun(const edm::Run&,
                                                                                 const edm::EventSetup& iES) const {
  edm::ESHandle<edm::eventsetup::test::DummyData> h;
  iES.getData(h);
  *h;
  return {};
}
void TestESDummyDataNoPrefetchAnalyzer::analyze(const edm::Event&, const edm::EventSetup& iES) {
  edm::ESHandle<edm::eventsetup::test::DummyData> h;
  iES.getData(h);
  *h;
}

DEFINE_FWK_MODULE(TestESDummyDataAnalyzer);
DEFINE_FWK_MODULE(TestESDummyDataNoPrefetchAnalyzer);
