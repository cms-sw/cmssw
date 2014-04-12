// -*- C++ -*-
//
// Package:    TestBxOrConfig
// Class:      TestBxOrConfig
// 
/**\class TestBxOrConfig TestBxOrConfig.cc L1TriggerConfig/TestBxOrConfig/src/TestBxOrConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"



#include "CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"

//
// class decleration
//

class TestBxOrConfig : public edm::EDAnalyzer {
   public:
      explicit TestBxOrConfig(const edm::ParameterSet&);
      ~TestBxOrConfig();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

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
TestBxOrConfig::TestBxOrConfig(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


TestBxOrConfig::~TestBxOrConfig()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TestBxOrConfig::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   edm::ESHandle<L1RPCBxOrConfig> bxOrConfig;
   iSetup.get<L1RPCBxOrConfigRcd>().get(bxOrConfig);

   std::cout << "Checking BX Or settings" << std::endl;

   std::cout<< "First BX : "<<bxOrConfig->getFirstBX()<<", Last BX : "<<bxOrConfig->getLastBX()<<std::endl;

}


// ------------ method called once each job just before starting event loop  ------------
void 
TestBxOrConfig::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestBxOrConfig::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestBxOrConfig);
