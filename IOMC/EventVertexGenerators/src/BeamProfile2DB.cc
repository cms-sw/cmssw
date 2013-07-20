// -*- C++ -*-
//
// Package:    BeamProfile2DB
// Class:      BeamProfile2DB
// 
/**\class BeamProfile2DB BeamProfile2DB.cc IOMC/BeamProfile2DB/src/BeamProfile2DB.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Fri Jan  6 14:49:42 CET 2012
// $Id: BeamProfile2DB.cc,v 1.2 2013/02/27 18:41:07 wmtan Exp $
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

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"


//
// class declaration
//

class BeamProfile2DB : public edm::EDAnalyzer {
   public:
      explicit BeamProfile2DB(const edm::ParameterSet&);
      ~BeamProfile2DB();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      // ----------member data ---------------------------
  edm::ParameterSet config_;
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
BeamProfile2DB::BeamProfile2DB(const edm::ParameterSet& iConfig)

{
  config_=iConfig;  
}


BeamProfile2DB::~BeamProfile2DB()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
BeamProfile2DB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
BeamProfile2DB::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BeamProfile2DB::endJob() 
{
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  SimBeamSpotObjects * beam = new SimBeamSpotObjects();
  
  beam->read(config_);
  
  poolDbService->createNewIOV<SimBeamSpotObjects>(beam,
						  poolDbService->beginOfTime(),poolDbService->endOfTime(),
						  "SimBeamSpotObjectsRcd"  );

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
BeamProfile2DB::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamProfile2DB);
