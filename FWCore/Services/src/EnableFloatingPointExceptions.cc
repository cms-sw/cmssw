// -*- C++ -*-
//
// Package:     Services
// Class  :     EnableFloatingPointExceptions
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr 12 09:27:27 EDT 2006
// $Id$
//

// system include files
#include <iostream>
#if defined(__linux__)
#include <fenv.h>
#endif

// user include files
#include "FWCore/Services/src/EnableFloatingPointExceptions.h"
#include "FWCore/Utilities/interface/EDMException.h"

//
// constants, enums and typedefs
//
using namespace edm::service;
//
// static data member definitions
//

static float divideByZero(float zero)
{
  float x = 1.0/zero;
  
  return x;
}

//
// constructors and destructor
//
EnableFloatingPointExceptions::EnableFloatingPointExceptions(const edm::ParameterSet& iPSet)
#if defined(__linux__)
:initialMask_(fegetexcept() )
#endif
{
  enable(true);
  
  if(iPSet.getUntrackedParameter("runTest",false)) {
    float y = divideByZero(0.0);
    throw edm::Exception(edm::errors::LogicError) <<"SIGFPE was not activated."
      <<"When did test of a divide by zero we get the answer"<<y<<"\n  Please send email to the framework developers";
  }
}

// EnableFloatingPointExceptions::EnableFloatingPointExceptions(const EnableFloatingPointExceptions& rhs)
// {
//    // do actual copying here;
// }

EnableFloatingPointExceptions::~EnableFloatingPointExceptions()
{
  enable(false);
}

//
// assignment operators
//
// const EnableFloatingPointExceptions& EnableFloatingPointExceptions::operator=(const EnableFloatingPointExceptions& rhs)
// {
//   //An exception safe implementation is
//   EnableFloatingPointExceptions temp(rhs);
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
void
EnableFloatingPointExceptions::enable(bool on) const
{
#if defined(__linux__)
  const int kTurnOn = FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID;
  int mask = initialMask_;
  int omask;
  if( on ) {
    mask |= kTurnOn;
    omask = feenableexcept(mask);
  } else {
    omask = fedisableexcept(kTurnOn ^ initialMask_ );
  }
  //edm::LogInfo("FPESetting")
  //  << "FPE mask changed from " << std::hex << omask << " to " 
  //  << mask << std::dec;
#else
  //edm::LogWarning("FPESetting")
  //  << "FPEEnable is not supported on this platform.";
#endif
}
//
// static member functions
//
