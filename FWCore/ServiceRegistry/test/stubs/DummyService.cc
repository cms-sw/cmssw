// -*- C++ -*-
//
// Package:     test
// Class  :     DummyService
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:52:01 EDT 2005
//

// system include files

// user include files
#include "FWCore/ServiceRegistry/test/stubs/DummyService.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constants, enums and typedefs
//
using namespace testserviceregistry;

//
// static data member definitions
//

void
DummyService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

    edm::ParameterSetDescription desc;

    desc.add<int>("value");

    descriptions.addDefault(desc);
}

//
// constructors and destructor
//
DummyService::DummyService(const edm::ParameterSet& iPSet,edm::ActivityRegistry&iAR):
value_(iPSet.getParameter<int>("value")),
beginJobCalled_(false)
{
   iAR.watchPostBeginJob(this,&DummyService::doOnBeginJob);
}

// DummyService::DummyService(const DummyService& rhs)
// {
//    // do actual copying here;
// }

DummyService::~DummyService()
{
}

//
// assignment operators
//
// const DummyService& DummyService::operator=(const DummyService& rhs)
// {
//   //An exception safe implementation is
//   DummyService temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
