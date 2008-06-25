// -*- C++ -*-
//
// Package:    TestFailuresAnalyzer
// Class:      TestFailuresAnalyzer
//
/**\class TestFailuresAnalyzer TestFailuresAnalyzer.cc stubs/TestFailuresAnalyzer/src/TestFailuresAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep  2 13:54:17 EDT 2005
// $Id: TestFailuresAnalyzer.cc,v 1.3 2007/08/07 22:34:20 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/test/stubs/TestFailuresAnalyzer.h"

#include "FWCore/Utilities/interface/Exception.h"

//
// class decleration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

enum {
   kConstructor,
   kBeginOfJob,
   kEvent,
   kEndOfJob,
   kBeginOfJobBadXML
};
//
// constructors and destructor
//
TestFailuresAnalyzer::TestFailuresAnalyzer(const edm::ParameterSet& iConfig)
: whichFailure_(iConfig.getParameter<int>("whichFailure"))
{
   //now do what ever initialization is needed
   if(whichFailure_ == kConstructor){
      throw cms::Exception("Test")<<" constructor";
   }
}


TestFailuresAnalyzer::~TestFailuresAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TestFailuresAnalyzer::beginJob(const edm::EventSetup&)
{
   if(whichFailure_ == kBeginOfJob){
      throw cms::Exception("Test") <<" beginJob";
   }
   if(whichFailure_ == kBeginOfJobBadXML){
      throw cms::Exception("Test") <<" beginJob with <BAD> >XML<";
   }
}

void
TestFailuresAnalyzer::endJob()
{
   if(whichFailure_ == kEndOfJob){
      throw cms::Exception("Test") <<" endJob";
   }
}


void
TestFailuresAnalyzer::analyze(const edm::Event& /* iEvent */, const edm::EventSetup& /* iSetup */)
{
   if(whichFailure_ == kEvent){
      throw cms::Exception("Test") <<" event";
   }

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestFailuresAnalyzer);
