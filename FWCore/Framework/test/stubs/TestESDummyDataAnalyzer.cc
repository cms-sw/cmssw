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
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/test/DummyData.h"

#include "FWCore/Utilities/interface/Exception.h"

class TestESDummyDataAnalyzer : public edm::EDAnalyzer {
public:
  explicit TestESDummyDataAnalyzer(const edm::ParameterSet&);

private:
  virtual void endJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  int m_expectedValue;
  int const m_nEventsValue;
  int m_counter{};
  int m_totalCounter{};
  int const m_totalNEvents;
  edm::ESGetTokenT<edm::eventsetup::test::DummyData> const m_esToken;
};

TestESDummyDataAnalyzer::TestESDummyDataAnalyzer(const edm::ParameterSet& iConfig) :
  m_expectedValue{iConfig.getParameter<int>("expected")},
  m_nEventsValue{iConfig.getUntrackedParameter<int>("nEvents",0)},
  m_totalNEvents{iConfig.getUntrackedParameter<int>("totalNEvents",-1)},
  m_esToken{esConsumes<edm::eventsetup::test::DummyData, edm::DefaultRecord>()}
{}


void
TestESDummyDataAnalyzer::analyze(const edm::Event&, const edm::EventSetup& iSetup)
{
  using namespace edm;

  ++m_totalCounter;
  if(m_nEventsValue) {
    ++m_counter;
    if(m_nEventsValue<m_counter) {
      ++m_expectedValue;
      m_counter=0;
    }
  }

  ESHandle<edm::eventsetup::test::DummyData> pData;
  iSetup.getData(m_esToken, pData);

  if(m_expectedValue != pData->value_) {
    throw cms::Exception("WrongValue")<<"got value "<<pData->value_<<" but expected "<<m_expectedValue;
  }

}

void
TestESDummyDataAnalyzer::endJob()
{
  if (-1 != m_totalNEvents &&
      m_totalNEvents != m_totalCounter) {
    throw cms::Exception("WrongNumberOfEvents")<<"expected "<<m_totalNEvents<<" but instead saw "<<m_totalCounter
                                               <<"\n";
  }
}

DEFINE_FWK_MODULE(TestESDummyDataAnalyzer);
