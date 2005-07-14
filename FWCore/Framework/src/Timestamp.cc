// -*- C++ -*-
//
// Package:     Framework
// Module:      Timestamp
// 
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 24 16:27:03 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/Timestamp.h"

namespace edm {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
   Timestamp::Timestamp(unsigned long iValue) : time_(iValue)
{
}

// Timestamp::Timestamp(const Timestamp& rhs)
// {
//    // do actual copying here;
// }

Timestamp::~Timestamp()
{
}

//
// assignment operators
//
// const Timestamp& Timestamp::operator=(const Timestamp& rhs)
// {
//   //An exception safe implementation is
//   Timestamp temp(rhs);
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
const Timestamp&
Timestamp::invalidTimestamp() {
   static Timestamp s_invalid(0);
   return s_invalid;
}
const Timestamp&
Timestamp::endOfTime() {
   static Timestamp s_endOfTime(0xFFFFFFFFUL);
   return s_endOfTime;
}
const Timestamp&
Timestamp::beginOfTime() {
   static Timestamp s_beginOfTime(1);
   return s_beginOfTime;
}

}
