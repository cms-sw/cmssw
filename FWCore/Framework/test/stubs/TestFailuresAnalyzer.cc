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
//
//

// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/test/stubs/TestFailuresAnalyzer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
//
// class decleration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

enum { kConstructor, kBeginOfJob, kEvent, kEndOfJob, kBeginOfJobBadXML, kEventCorruption };
//
// constructors and destructor
//
TestFailuresAnalyzer::TestFailuresAnalyzer(const edm::ParameterSet& iConfig)
    : whichFailure_(iConfig.getParameter<int>("whichFailure")),
      eventToThrow_(iConfig.getUntrackedParameter<unsigned long long>("eventToThrow", 2U)) {
  //now do what ever initialization is needed
  if (whichFailure_ == kConstructor) {
    throw cms::Exception("Test") << " constructor";
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TestFailuresAnalyzer::beginJob() {
  if (whichFailure_ == kBeginOfJob) {
    throw cms::Exception("Test") << " beginJob";
  }
  if (whichFailure_ == kBeginOfJobBadXML) {
    throw cms::Exception("Test") << " beginJob with <BAD> >XML<";
  }
}

void TestFailuresAnalyzer::endJob() {
  if (whichFailure_ == kEndOfJob) {
    throw cms::Exception("Test") << " endJob";
  }
}

void TestFailuresAnalyzer::analyze(edm::StreamID,
                                   const edm::Event& e /* iEvent */,
                                   const edm::EventSetup& /* iSetup */) const {
  if (whichFailure_ == kEvent) {
    throw cms::Exception("Test") << " event";
  }
  if (whichFailure_ == kEventCorruption && eventToThrow_ == e.eventAuxiliary().event()) {
    throw edm::Exception(edm::errors::EventCorruption, "testing exception handling");
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestFailuresAnalyzer);
