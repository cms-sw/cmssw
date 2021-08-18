// -*- C++ -*-
//
// Package:    TestModuleChangeLooper
// Class:      TestModuleChangeLooper
//
/**\class TestModuleChangeLooper TestModuleChangeLooper.h FWCore/TestModuleChangeLooper/interface/TestModuleChangeLooper.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Valentin Kuznetsov
//         Created:  Tue Jul 18 10:17:05 EDT 2006
//

// user include files
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/ModuleChanger.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"

// system include files
#include <memory>

//
// class decleration
//

class TestModuleChangeLooper : public edm::EDLooper {
public:
  TestModuleChangeLooper(edm::ParameterSet const&);
  ~TestModuleChangeLooper();

  void startingNewLoop(unsigned int) {}
  Status duringLoop(edm::Event const& iEvent, edm::EventSetup const&) {
    auto const& product = iEvent.get(m_token);
    if (product.value != m_expectedValue) {
      throw cms::Exception("WrongValue") << "expected value " << m_expectedValue << " but got " << product.value;
    }
    return kContinue;
  }
  Status endOfLoop(edm::EventSetup const&, unsigned int iCount) {
    //modify the module
    Labels labels;
    labelsForToken(m_token, labels);
    std::string const moduleLabel{labels.module};

    edm::ParameterSet const* pset = scheduleInfo()->parametersForModule(moduleLabel);
    assert(0 != pset);

    edm::ParameterSet newPSet(*pset);
    newPSet.addParameter<int>("ivalue", ++m_expectedValue);
    auto success = moduleChanger()->changeModule(moduleLabel, newPSet);
    assert(success && "moduleChanger()->changeModule(m_tag.label(), newPSet)");

    return iCount == 2 ? kStop : kContinue;
  }

private:
  // ----------member data ---------------------------
  int m_expectedValue;
  edm::EDGetTokenT<edmtest::IntProduct> m_token;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TestModuleChangeLooper::TestModuleChangeLooper(edm::ParameterSet const& iConfig)
    : m_expectedValue(iConfig.getUntrackedParameter<int>("startingValue")),
      m_token(consumes(iConfig.getUntrackedParameter<edm::InputTag>("tag"))) {
  //now do what ever other initialization is needed
}

TestModuleChangeLooper::~TestModuleChangeLooper() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------

//define this as a plug-in
DEFINE_FWK_LOOPER(TestModuleChangeLooper);
