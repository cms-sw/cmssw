// -*- C++ -*-
//
// Package:    DumpL1RPCHsbConfig
// Class:      DumpL1RPCHsbConfig
// 
/**\class DumpL1RPCHsbConfig DumpL1RPCHsbConfig.cc L1TriggerConfig/DumpL1RPCHsbConfig/src/DumpL1RPCHsbConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/


// system include files
#include <memory>
#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"




#include <fstream>

//
// class decleration
//

class DumpL1RPCHsbConfig : public edm::EDAnalyzer {
   public:
      explicit DumpL1RPCHsbConfig(const edm::ParameterSet&);
      ~DumpL1RPCHsbConfig();


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
DumpL1RPCHsbConfig::DumpL1RPCHsbConfig(const edm::ParameterSet& iConfig)


{
   //now do what ever initialization is needed


}


DumpL1RPCHsbConfig::~DumpL1RPCHsbConfig()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DumpL1RPCHsbConfig::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   edm::ESHandle<L1RPCHsbConfig> hsbConfig;
   iSetup.get<L1RPCHsbConfigRcd>().get(hsbConfig);
   
   LogDebug("DumpL1RPCHsbConfig")<< "Checking HSB inputs" << std::endl;
  
   std::vector<int> mask0 = hsbConfig->getHsb0Mask();
   std::vector<int> mask1 = hsbConfig->getHsb1Mask(); 

   LogDebug("DumpL1RPCHsbConfig") << " HSB0: ";
   for (unsigned int i = 0; i < mask0.size() ; i++){
     std::cout<<mask0[i]<<" ";
   }
   std::cout<<std::endl;
   LogDebug("DumpL1RPCHsbConfig") << " HSB1: ";
   for (unsigned int i = 0; i < mask1.size() ; i++){
     std::cout<<mask1[i]<<" ";
   }
   std::cout<<std::endl;
   
}


// ------------ method called once each job just before starting event loop  ------------
void 
DumpL1RPCHsbConfig::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DumpL1RPCHsbConfig::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpL1RPCHsbConfig);
