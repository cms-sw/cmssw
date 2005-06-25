// -*- C++ -*-
//
// Package:    WhatsItAnalyzer
// Class:      WhatsItAnalyzer
// 
/**\class WhatsItAnalyzer WhatsItAnalyzer.cc test/WhatsItAnalyzer/src/WhatsItAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 19:13:25 EDT 2005
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"
#include "FWCore/CoreFramework/interface/EDAnalyzer.h"

#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/FWCoreIntegration/src/WhatsIt.h"
#include "FWCore/FWCoreIntegration/src/GadgetRcd.h"

#include "FWCore/CoreFramework/interface/ESHandle.h"
#include "FWCore/CoreFramework/interface/EventSetup.h"
//
// class decleration
//

namespace edmreftest {

class WhatsItAnalyzer : public edm::EDAnalyzer {
   public:
      explicit WhatsItAnalyzer( const edm::ParameterSet& );
      ~WhatsItAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
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
WhatsItAnalyzer::WhatsItAnalyzer( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed

}


WhatsItAnalyzer::~WhatsItAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
WhatsItAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;
   eventsetup::ESHandle<WhatsIt> pSetup;
   iSetup.get<GadgetRcd>().get( pSetup );
}

}
using namespace edmreftest;
//define this as a plug-in
DEFINE_FWK_MODULE(WhatsItAnalyzer)
