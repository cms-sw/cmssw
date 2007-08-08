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
// $Id: WhatsItAnalyzer.cc,v 1.9 2006/10/21 16:44:13 wmtan Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/Integration/test/WhatsIt.h"
#include "FWCore/Integration/test/GadgetRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
//
// class decleration
//

namespace edmtest {

class WhatsItAnalyzer : public edm::EDAnalyzer {
   public:
      explicit WhatsItAnalyzer(const edm::ParameterSet&);
      ~WhatsItAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
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
WhatsItAnalyzer::WhatsItAnalyzer(const edm::ParameterSet& /*iConfig*/)
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
WhatsItAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup)
{
   using namespace edm;
   ESHandle<WhatsIt> pSetup;
   iSetup.get<GadgetRcd>().get(pSetup);

   std::cout <<"WhatsIt "<<pSetup->a<<std::endl;
}

}
using namespace edmtest;
//define this as a plug-in
DEFINE_FWK_MODULE(WhatsItAnalyzer);
