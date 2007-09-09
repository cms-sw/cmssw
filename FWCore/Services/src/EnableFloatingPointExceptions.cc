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
// $Id: EnableFloatingPointExceptions.cc,v 1.12 2007/06/14 21:03:39 wmtan Exp $
//

// system include files

#ifdef __linux__
#ifdef __i386__
#include <fpu_control.h>
#endif
#endif

// user include files
#include "FWCore/Services/src/EnableFloatingPointExceptions.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm::service;
//
// constants, enums and typedefs
//

//
// constructors and destructor
//
EnableFloatingPointExceptions::EnableFloatingPointExceptions(const ParameterSet& iPS, ActivityRegistry&iRegistry):
enableDivByZeroEx_(false),
enableInvalidEx_(false),
enableOverFlowEx_(false),
enableUnderFlowEx_(false),
setPrecisionDouble_(true),
reportSettings_(false)
{
  iRegistry.watchPostEndJob(this,&EnableFloatingPointExceptions::postEndJob);
  iRegistry.watchPreModule(this, &EnableFloatingPointExceptions::preModule);
  iRegistry.watchPostModule(this, &EnableFloatingPointExceptions::postModule);

  reportSettings_     = iPS.getUntrackedParameter<bool>("reportSettings",false);
  setPrecisionDouble_ = iPS.getUntrackedParameter<bool>("setPrecisionDouble",true);

// Get the state of the fpu and save it as the "OSdefault" state. The language here
// is a bit odd.  We use "OSdefault" to label the fpu state we inherit from the OS on
// job startup.  By contrast, "default" is the label we use for the fpu state when either
// we or the user has specified a state for modules not appearing in the module list.
// Generally, "OSdefault" and "default" are the same but are not required to be so.

  fegetenv( &fpuState_ );
  OSdefault_ = fpuState_;
  stateStack_.push(OSdefault_);
  if( reportSettings_ )  {
    edm::LogVerbatim("FPE_Enable") << "\nSettings for OSdefault";
    echoState();
  }

// Then go handle the specific cases as described in the cfg file

  PSet    empty_PSet;
  VString empty_VString;

  VString moduleNames = iPS.getUntrackedParameter<VString>("moduleNames",empty_VString);

// If the module name list is missing or empty, set default values for all parameters

  if( moduleNames.empty() ) {
    enableDivByZeroEx_  = false;
    enableInvalidEx_    = false;
    enableOverFlowEx_   = false;
    enableUnderFlowEx_  = false;
    setPrecisionDouble_ = true;

    controlFpe();
    if( reportSettings_ ) {
      edm::LogVerbatim("FPE_Enable") << "\nSettings for default";
      echoState();
    }
    fegetenv( &fpuState_ );
    stateMap_["default"] =  fpuState_;
  } else {

// Otherwise, scan the module name list and set per-module values.  Be careful to treat
// any user-specified default first.  If there is one, use it to override our default.
// Then remove it from the list so we don't see it again while handling everything else.

    VString::iterator pos = find(moduleNames.begin(), moduleNames.end(), "default");
    if( pos != moduleNames.end() ) {
      PSet secondary = iPS.getUntrackedParameter<PSet>(*pos, empty_PSet);
      enableDivByZeroEx_  = secondary.getUntrackedParameter<bool>("enableDivByZeroEx", false);
      enableInvalidEx_    = secondary.getUntrackedParameter<bool>("enableInvalidEx",   false);
      enableOverFlowEx_   = secondary.getUntrackedParameter<bool>("enableOverFlowEx",  false);
      enableUnderFlowEx_  = secondary.getUntrackedParameter<bool>("enableUnderFlowEx", false);
      controlFpe();
      if( reportSettings_ ) {
        edm::LogVerbatim("FPE_Enable") << "\nSettings for unnamed module";
        echoState();
      }
      fegetenv( &fpuState_ );
      stateMap_["default"] =  fpuState_;
      moduleNames.erase(pos);
    }        

// Then handle the rest.

    for( VString::const_iterator it(moduleNames.begin()), itEnd=moduleNames.end(); it != itEnd; ++it) {
      PSet secondary = iPS.getUntrackedParameter<PSet>(*it, empty_PSet);
      enableDivByZeroEx_  = secondary.getUntrackedParameter<bool>("enableDivByZeroEx", false);
      enableInvalidEx_    = secondary.getUntrackedParameter<bool>("enableInvalidEx",   false);
      enableOverFlowEx_   = secondary.getUntrackedParameter<bool>("enableOverFlowEx",  false);
      enableUnderFlowEx_  = secondary.getUntrackedParameter<bool>("enableUnderFlowEx", false);
      controlFpe();
      if( reportSettings_ ) {
        edm::LogVerbatim("FPE_Enable") << "\nSettings for module " << *it;
        echoState();
      }
      fegetenv( &fpuState_ );
      stateMap_[*it] =  fpuState_;
    }
  }

// And finally, put the state back to the way we found it originally

    fesetenv( &OSdefault_ );
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
EnableFloatingPointExceptions::postEndJob()
{

// At EndJob, put the state of the fpu back to "OSdefault"

  fpuState_ = stateMap_[String("OSdefault")];
  fesetenv( &OSdefault_ );
  if( reportSettings_ ) edm::LogVerbatim("FPE_Enable") << "\nSettings at end job ";
  echoState();
}

void 
EnableFloatingPointExceptions::preModule(const ModuleDescription& iDescription)
{

// On entry to a module, find the desired state of the fpu and set it accordingly.
// Note that any module whose label does not appear in our list gets the default settings.

  String modName = iDescription.moduleLabel();
  if( stateMap_.find(modName) == stateMap_.end() )  {
    fpuState_ = stateMap_[String("default")];
  } else {
    fpuState_ = stateMap_[modName];
  }
  fesetenv( &fpuState_ );
  stateStack_.push(fpuState_);
  if( reportSettings_ ) {
    edm::LogVerbatim("FPE_Enable") << "\nSettings at begin module " << modName;
    echoState();
  }
}
void 
EnableFloatingPointExceptions::postModule(const ModuleDescription& iDescription)
{

// On exit from a module, set the state of the fpu back to what it was before entry

  stateStack_.pop();
  fpuState_ = stateStack_.top();
  fesetenv( &fpuState_ );
  if( reportSettings_ ) {
    edm::LogVerbatim("FPE_Enable") << "\nSettings after end module ";
    echoState();
  }
}

void 
EnableFloatingPointExceptions::controlFpe()
{
  // Local Declarations

  unsigned short int FE_PRECISION = 1<<5;
  unsigned short int suppress;

#ifdef __linux__

/*
 * NB: We are not letting users control signaling inexact (FE_INEXACT).
 */

  suppress = FE_PRECISION;
  if ( !enableDivByZeroEx_ ) suppress |= FE_DIVBYZERO;
  if ( !enableInvalidEx_ )   suppress |= FE_INVALID;
  if ( !enableOverFlowEx_ )  suppress |= FE_OVERFLOW;
  if ( !enableUnderFlowEx_ ) suppress |= FE_UNDERFLOW;
  fegetenv( &fpuState_ );
  fpuState_.__control_word = suppress;
  fesetenv( &fpuState_ );

#ifdef __i386__

  if (setPrecisionDouble_) {
    fpu_control_t cw;
    _FPU_GETCW(cw);

    cw = (cw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
    _FPU_SETCW(cw);
  }
#endif
#endif

}

void
EnableFloatingPointExceptions::echoState()
{
  if( reportSettings_ ) {
    int femask = fegetexcept();
    edm::LogVerbatim("FPE_Enable") << "Floating point exception mask is " 
                                   << std::showbase << std::hex << femask;

    if( femask & FE_DIVBYZERO )
      edm::LogVerbatim("FPE_Enable") << "\tDivByZero exception is on";
    else
      edm::LogVerbatim("FPE_Enable") << "\tDivByZero exception is off";

    if( femask & FE_INVALID )
      edm::LogVerbatim("FPE_Enable") << "\tInvalid exception is on";
    else
      edm::LogVerbatim("FPE_Enable") << "\tInvalid exception is off";

    if( femask & FE_OVERFLOW )
      edm::LogVerbatim("FPE_Enable") << "\tOverFlow exception is on";
    else
      edm::LogVerbatim("FPE_Enable") << "\tOverflow exception is off";

    if( femask & FE_UNDERFLOW )
      edm::LogVerbatim("FPE_Enable") << "\tUnderFlow exception is on";
    else
      edm::LogVerbatim("FPE_Enable") << "\tUnderFlow exception is off";
  }
}
