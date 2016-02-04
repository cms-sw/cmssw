// -*- C++ -*-
//
// Package:    Integration
// Class:      UseValueExampleAnalyzer
// 
/**\class UseValueExampleAnalyzer UseValueExampleAnalyzer.cc FWCore/Integration/test/UseValueExampleAnalyzer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Chris D Jones
//         Created:  Thu Sep  8 03:55:42 EDT 2005
// $Id: UseValueExampleAnalyzer.cc,v 1.4 2007/08/08 16:44:49 wmtan Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/Integration/test/ValueExample.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class decleration
//

class UseValueExampleAnalyzer : public edm::EDAnalyzer {
public:
   explicit UseValueExampleAnalyzer(const edm::ParameterSet&);
   ~UseValueExampleAnalyzer();
   
   
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
UseValueExampleAnalyzer::UseValueExampleAnalyzer(const edm::ParameterSet& /* iConfig */)
{
   //now do what ever initialization is needed
   
}


UseValueExampleAnalyzer::~UseValueExampleAnalyzer()
{
   
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
UseValueExampleAnalyzer::analyze(const edm::Event& /* iEvent */, const edm::EventSetup& /* iSetup*/)
{   
   std::cout<<" value from service "<< edm::Service<ValueExample>()->value()<<std::endl; 
}

//define this as a plug-in
DEFINE_FWK_MODULE(UseValueExampleAnalyzer);

