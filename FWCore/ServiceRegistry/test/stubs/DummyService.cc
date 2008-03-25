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
// $Id: DummyService.cc,v 1.2 2005/09/10 02:08:49 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/ServiceRegistry/test/stubs/DummyService.h"

//
// constants, enums and typedefs
//
using namespace testserviceregistry;

//
// static data member definitions
//

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
