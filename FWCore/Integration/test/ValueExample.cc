// -*- C++ -*-
//
// Package:     test
// Class  :     ValueExample
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:52:01 EDT 2005
// $Id: ValueExample.cc,v 1.2 2005/09/10 02:08:46 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Integration/test/ValueExample.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ValueExample::ValueExample(const edm::ParameterSet& iPSet):
value_(iPSet.getParameter<int>("value"))
{
}

// ValueExample::ValueExample(const ValueExample& rhs)
// {
//    // do actual copying here;
// }

ValueExample::~ValueExample()
{
}

//
// assignment operators
//
// const ValueExample& ValueExample::operator=(const ValueExample& rhs)
// {
//   //An exception safe implementation is
//   ValueExample temp(rhs);
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
using namespace edm::serviceregistry;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE_MAKER(ValueExample,ParameterSetMaker<ValueExample>);
