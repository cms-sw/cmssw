// -*- C++ -*-
//
// Package:     test
// Class  :     DependsOnDummyService
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:52:01 EDT 2005
// $Id$
//

// system include files

// user include files
#include "FWCore/ServiceRegistry/test/stubs/DependsOnDummyService.h"
#include "FWCore/ServiceRegistry/test/stubs/DummyService.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

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
DependsOnDummyService::DependsOnDummyService():
value_( edm::Service<DummyService>()->value() )
{
}

// DependsOnDummyService::DependsOnDummyService(const DependsOnDummyService& rhs)
// {
//    // do actual copying here;
// }

DependsOnDummyService::~DependsOnDummyService()
{
}

//
// assignment operators
//
// const DependsOnDummyService& DependsOnDummyService::operator=(const DependsOnDummyService& rhs)
// {
//   //An exception safe implementation is
//   DependsOnDummyService temp(rhs);
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
