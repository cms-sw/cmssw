// -*- C++ -*-
//
// Package:    TestHWConfig
// Class:      TestHWConfig
// 
/**\class TestHWConfig TestHWConfig.cc L1TriggerConfig/TestHWConfig/src/TestHWConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Apr  9 14:03:40 CEST 2008
// $Id: TestHWConfig.cc,v 1.5 2010/02/26 15:51:02 fruboes Exp $
//
//


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



#include "CondFormats/DataRecord/interface/L1RPCHwConfigRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"

//
// class decleration
//

class TestHWConfig : public edm::EDAnalyzer {
   public:
      explicit TestHWConfig(const edm::ParameterSet&);
      ~TestHWConfig();


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
TestHWConfig::TestHWConfig(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


TestHWConfig::~TestHWConfig()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TestHWConfig::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   edm::ESHandle<L1RPCHwConfig> hwConfig;
   iSetup.get<L1RPCHwConfigRcd>().get(hwConfig);

   std::cout << "Checking crates " << std::endl;

   for (int crate = 0; crate < 12 ; ++crate){

    std::set<int> enabledTowers;

    for (int tw = -16; tw < 17 ; ++tw){

      if ( hwConfig->isActive(tw, crate,0 ) )
        enabledTowers.insert(tw);

    }

    if ( !enabledTowers.empty() ){
       std::cout << "Crate " << crate
                 << ", active towers:";

       std::set<int>::iterator it; 
       for (it=enabledTowers.begin();it!=enabledTowers.end(); ++it){
          std::cout << " " << *it;
       }
       std::cout << std::endl;

    } // printout

  }  // crate iteration ends

  //std::cout << "First BX: "<<hwConfig->getFirstBX()<<", last BX: "<<hwConfig->getLastBX()<<std::endl;

  std::cout << " Done " << hwConfig->size() << std::endl;
}


// ------------ method called once each job just before starting event loop  ------------
void 
TestHWConfig::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestHWConfig::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestHWConfig);
