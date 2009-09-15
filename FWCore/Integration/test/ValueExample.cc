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
//

// system include files

// user include files
#include "FWCore/Integration/test/ValueExample.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
