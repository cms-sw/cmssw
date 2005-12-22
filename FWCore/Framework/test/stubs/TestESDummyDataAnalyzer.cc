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
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/test/DummyData.h"

#include "FWCore/Utilities/interface/Exception.h"

//
// class decleration
//

class TestESDummyDataAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TestESDummyDataAnalyzer(const edm::ParameterSet&);
      ~TestESDummyDataAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
         int m_expectedValue;
      // ----------member data ---------------------------
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
TestESDummyDataAnalyzer::TestESDummyDataAnalyzer(const edm::ParameterSet& iConfig) :
m_expectedValue(iConfig.getParameter<int>("expected"))
{
   //now do what ever initialization is needed

}


TestESDummyDataAnalyzer::~TestESDummyDataAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TestESDummyDataAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   ESHandle<edm::eventsetup::test::DummyData> pData;
   iSetup.getData(pData);

   if(m_expectedValue != pData->value_) {
      throw cms::Exception("WrongValue")<<"got value "<<pData->value_<<" but expected "<<m_expectedValue;
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestESDummyDataAnalyzer)
