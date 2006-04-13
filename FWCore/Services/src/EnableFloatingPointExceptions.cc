// -*- C++ -*-
//
// Package:     Services
// Class  :     EnableFloatingPointExceptions
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  E. Sexton-Kennedy
//         Created:  Tue Apr 11 13:43:16 CDT 2006
//
// $Id: EnableFloatingPointExceptions.cc,v 1.2 2006/04/13 14:29:43 lsexton Exp $
//

// system include files
#ifdef __linux__
#include <features.h>
//#if defined(__GLIBC__)&&(__GLIBC__>2 || __GLIBC__==2 && __GLIBC_MINOR__>=1)
//#define _GNU_SOURCE
#include <fenv.h>
//#endif
#endif /* LINUX */

#ifdef SunOS
#include <ieeefp.h>
#include <assert.h>
#endif

#include <limits> // For testing

// user include files
#include "FWCore/Services/src/EnableFloatingPointExceptions.h"

#include "DataFormats/Common/interface/ModuleDescription.h"

using namespace edm::service;
//
// constants, enums and typedefs
//

//
// local static functions for testing
//
static float divideByZero(float zero)
{
  float x = 1.0/zero;
  
  return x;
}
static float useNan()
{
  // libstd++ lies and says that intel doesn't have one of these so we do it ourselves
  //float x = 1.0*std::numeric_limits<float>::quiet_NaN();
  
  //union { unsigned int u; float f; } aNaN;
  //aNaN.u = 0x7fc00000UL;
  //we found out that propagating nans doesn't cause the exception so we'll do something
  //that will generate a nan
  union { unsigned int u; float f; } inf;
  inf.u = 0x7f800000UL;
  float x = 0.0*inf.f;

  return x;
}
static float generateOverFlow()
{
  float x = 2.0*std::numeric_limits<float>::max();
  
  return x;
}
static float generateUnderFlow()
{
  float x = std::numeric_limits<float>::min()/2.0;
  
  return x;
}

//
// constructors and destructor
//
EnableFloatingPointExceptions::EnableFloatingPointExceptions(const ParameterSet& iPS, ActivityRegistry&iRegistry):
enableDivByZeroEx_(iPS.getUntrackedParameter<bool>("enableDivByZeroEx",true)),
enableInvalidEx_(iPS.getUntrackedParameter<bool>("enableInvalidEx",true)),
enableOverFlowEx_(iPS.getUntrackedParameter<bool>("enableOverFlowEx",true)),
enableUnderFlowEx_(iPS.getUntrackedParameter<bool>("enableUnderFlowEx",false))
{
  controlFpe();
  //iRegistry.watchPreModule(this,&EnableFloatingPointExceptions::preModule);
  //iRegistry.watchPostModule(this,&EnableFloatingPointExceptions::postModule);
  // Now run tests if requested
  if(iPS.getUntrackedParameter("runTest",false))
  {
    if(enableDivByZeroEx_)
    {
      float y = divideByZero(0.0);
      throw edm::Exception(edm::errors::LogicError) <<"SIGFPE was not activated."
       <<"While doing a test of divide by zero, we get the answer "<<y<<"\n  Please send email to the framework developers";
    }
    if(enableInvalidEx_)
    {
      float y = useNan();
      throw edm::Exception(edm::errors::LogicError) <<"SIGFPE was not activated."
      <<"While doing a test of invalid arguements we get the answer "<<y<<"\n  Please send email to the framework developers";
    }
    if(enableOverFlowEx_)
    {
      float y = generateOverFlow();
      throw edm::Exception(edm::errors::LogicError) <<"SIGFPE was not activated."
      <<"While doing a test of an overflow we get the answer "<<y<<"\n  Please send email to the framework developers";
    }
    if(enableUnderFlowEx_)
    {
      float y = generateUnderFlow();
      throw edm::Exception(edm::errors::LogicError) <<"SIGFPE was not activated."
      <<"While doing a test of an underflow we get the answer "<<y<<"\n  Please send email to the framework developers";
    }
  }
}

// EnableFloatingPointExceptions::EnableFloatingPointExceptions(const EnableFloatingPointExceptions& rhs)
// {
//    // do actual copying here;
// }

//EnableFloatingPointExceptions::~EnableFloatingPointExceptions()
//{
//}

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

void 
EnableFloatingPointExceptions::preModule(const ModuleDescription& iDescription)
{
  // For future dev.
}

void 
EnableFloatingPointExceptions::postModule(const ModuleDescription& iDescription)
{
  // For future dev
}
void 
EnableFloatingPointExceptions::controlFpe()
{
  // Local Declarations
#ifdef SunOS
	fp_except fpu_exceptions;
#endif
#ifdef __linux__

	/*
	 * NB: We are not letting users control signaling inexact (FE_INEXACT).
	 */
	if ( enableDivByZeroEx_ )
	  (void) feenableexcept( FE_DIVBYZERO );
	else 
	  (void) fedisableexcept( FE_DIVBYZERO );

	if ( enableInvalidEx_ )
	  (void) feenableexcept( FE_INVALID );
	else 
	  (void) fedisableexcept( FE_INVALID );

	if ( enableOverFlowEx_ )
	  (void) feenableexcept( FE_OVERFLOW );
	else 
	  (void) fedisableexcept( FE_OVERFLOW );

	if ( enableUnderFlowEx_ )
	  (void) feenableexcept( FE_UNDERFLOW );
	else 
	  (void) fedisableexcept( FE_UNDERFLOW );

#endif /* LINUX */

#ifdef SunOS
	/*
	 * NB: We are not letting users control signaling Imprecise (FP_X_IMP).
	 */

	if ( enableDivByZeroEx_ )
	  (void) fpsetmask( (fpu_exceptions | FP_X_DZ) );
	else 
	  (void) fpsetmask( (fpu_exceptions & ~FP_X_DZ) );

	if ( enableInvalidEx_ )
	  (void) fpsetmask( (fpu_exceptions | FP_X_INV) );
	else 
	  (void) fpsetmask( (fpu_exceptions & ~FP_X_INV) );

	if ( enableOverFlowEx_ )
	  (void) fpsetmask( (fpu_exceptions | FP_X_OFL) );
	else 
	  (void) fpsetmask( (fpu_exceptions & ~FP_X_OFL) );

	if ( enableUnderFlowEx_ )
	  (void) fpsetmask( (fpu_exceptions | FP_X_UFL) );
	else 
	  (void) fpsetmask( (fpu_exceptions & ~FP_X_UFL) );

#endif  /* SunOS */
}
//
// const member functions
//
