// -*- C++ -*-
//
// Package:    RPCDetIdAnalyzer
// Class:      RPCDetIdAnalyzer
// 
/**\class RPCDetIdAnalyzer RPCDetIdAnalyzer.cc DataFormats/RPCDetIdAnalyzer/src/RPCDetIdAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marcello Maggi,161 R-006,+41227676292,
//         Created:  Fri Nov  4 12:32:59 CET 2011
// $Id: RPCDetIdAnalyzer.cc,v 1.1 2011/11/05 10:39:54 mmaggi Exp $
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
#include "DataFormats/MuonDetId/interface/RPCCompDetId.h"
//
// class declaration
//

class RPCDetIdAnalyzer : public edm::EDAnalyzer {
   public:
      explicit RPCDetIdAnalyzer(const edm::ParameterSet&);
      ~RPCDetIdAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

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
RPCDetIdAnalyzer::RPCDetIdAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


RPCDetIdAnalyzer::~RPCDetIdAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
RPCDetIdAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)

{
  RPCCompDetId rpcgasid("WM2_S04_RB4R",0);
  
  //RPCCompDetId rpcgasid("cms_rpc_dcs_03:EM1_R03_C12_C17_UP",0);
  std::cout <<rpcgasid<<" rawid = "<< rpcgasid.rawId()<<std::endl;
  std::cout <<"Region = "<<rpcgasid.region()<<std::endl;
  std::cout <<"Ring or Wheel = "<<rpcgasid.ring()<<" - Wheel = "<<rpcgasid.wheel()<<std::endl;
  std::cout <<"Station or Disk = "<<rpcgasid.station()<<" - Disk = "<<rpcgasid.disk()<<std::endl;
  std::cout <<"Sector = "<<rpcgasid.sector()<<std::endl;
  std::cout <<"Layer = "<<rpcgasid.layer()<<std::endl;
  std::cout <<"SubSector = "<<rpcgasid.subsector()<<std::endl;
  std::cout <<std::setw(100)<<std::setfill('-')<<std::endl;
  std::cout <<"ok"<<std::endl;
  RPCCompDetId check(rpcgasid.rawId());
  std::cout <<check<<" rawid = "<< check.rawId()<<std::endl;
  std::cout <<"Region = "<<check.region()<<std::endl;
  std::cout <<"Ring or Wheel = "<<check.ring()<<" - Wheel = "<<check.wheel()<<std::endl;
  std::cout <<"Station or Disk = "<<check.station()<<" - Disk = "<<check.disk()<<std::endl;
  std::cout <<"Sector = "<<check.sector()<<std::endl;
  std::cout <<"Layer = "<<check.layer()<<std::endl;
  std::cout <<"SubSector = "<<check.subsector()<<std::endl;
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
RPCDetIdAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RPCDetIdAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
RPCDetIdAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
RPCDetIdAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
RPCDetIdAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
RPCDetIdAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
RPCDetIdAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCDetIdAnalyzer);
