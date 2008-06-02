// -*- C++ -*-
//
// Package:    TestBeginEndJobAnalyzer
// Class:      TestBeginEndJobAnalyzer
// 
/**\class TestBeginEndJobAnalyzer TestBeginEndJobAnalyzer.cc stubs/TestBeginEndJobAnalyzer/src/TestBeginEndJobAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep  2 13:54:17 EDT 2005
// $Id: TestBeginEndJobAnalyzer.cc,v 1.5 2007/04/13 19:12:38 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"



#include "FWCore/Framework/test/stubs/TestBeginEndJobAnalyzer.h"

//
// class decleration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TestBeginEndJobAnalyzer::TestBeginEndJobAnalyzer(const edm::ParameterSet& /* iConfig */)
{
   //now do what ever initialization is needed

}


TestBeginEndJobAnalyzer::~TestBeginEndJobAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   destructorCalled = true;

}


//
// member functions
//
bool
TestBeginEndJobAnalyzer::beginJobCalled = false;

bool
TestBeginEndJobAnalyzer::endJobCalled = false;

bool
TestBeginEndJobAnalyzer::destructorCalled = false;

// ------------ method called to produce the data  ------------
void 
TestBeginEndJobAnalyzer::beginJob(const edm::EventSetup&)
{
   beginJobCalled = true;
}

void 
TestBeginEndJobAnalyzer::endJob()
{
   endJobCalled = true;
}


void
TestBeginEndJobAnalyzer::analyze(const edm::Event& /* iEvent */, const edm::EventSetup& /* iSetup */)
{
}
