// -*- C++ -*-
//
// Package:     EDProduct
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
#include "DataFormats/Provenance/interface/Timestamp.h"

namespace edm {
//
// constants, enums and typedefs
//

//
// static data member definitions
//
static const TimeValue_t kLowMask(0xFFFFFFFF);
   
//
// constructors and destructor
//
Timestamp::Timestamp(TimeValue_t iValue) : 
   timeLow_(static_cast<unsigned int>(kLowMask & iValue)),
   timeHigh_(static_cast<unsigned int>(iValue >> 32))
{
}

Timestamp::Timestamp() : timeLow_(invalidTimestamp().timeLow_),
timeHigh_(invalidTimestamp().timeHigh_)
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
TimeValue_t 
Timestamp::value() const {
   TimeValue_t returnValue = timeHigh_;
   returnValue = returnValue << 32;
   returnValue += timeLow_;
   return returnValue;
}

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
   //calculate 2^N -1 where N is number of bits in TimeValue_t
   // by doing 2^(N-1) - 1 + 2^(N-1)
   const TimeValue_t temp = TimeValue_t(1) << (sizeof(TimeValue_t)/sizeof(char) * 8 - 1);
   static Timestamp s_endOfTime((temp -1) + temp);
   return s_endOfTime;
}
const Timestamp&
Timestamp::beginOfTime() {
   static Timestamp s_beginOfTime(1);
   return s_beginOfTime;
}

}
