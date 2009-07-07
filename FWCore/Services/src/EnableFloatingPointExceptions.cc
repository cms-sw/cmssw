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
//#include "FWCore/Utilities/interface/Algorithms.h"

using namespace edm::service;
//
// constants, enums and typedefs
//

//
// constructors and destructor
//
EnableFloatingPointExceptions::EnableFloatingPointExceptions(ParameterSet const& iPS, ActivityRegistry&iRegistry):
  
  reportSettings_(false),
  fpuState_(),
  defaultState_(),
  OSdefault_(),
  stateMap_(),
  stateStack_() {

  iRegistry.watchPostEndJob(this,&EnableFloatingPointExceptions::postEndJob);
  iRegistry.watchPreModule(this, &EnableFloatingPointExceptions::preModule);
  iRegistry.watchPostModule(this, &EnableFloatingPointExceptions::postModule);

  reportSettings_     = iPS.getUntrackedParameter<bool>("reportSettings",false);
  bool setPrecisionDouble = iPS.getUntrackedParameter<bool>("setPrecisionDouble",true);

  // Get the state of the fpu and save it as the "OSdefault" state. The language here
  // is a bit odd.  We use "OSdefault" to label the fpu state we inherit from the OS on
  // job startup.  By contrast, "default" is the label we use for the fpu state when either
  // we or the user has specified a state for modules not appearing in the module list.
  // Generally, "OSdefault" and "default" are the same but are not required to be so.

  fegetenv(&fpuState_);
  OSdefault_ = fpuState_;
  //stateStack_.push(OSdefault_);
  if(reportSettings_)  {
    edm::LogVerbatim("FPE_Enable") << "\nSettings for OSdefault";
    echoState();
  }

  // Then go handle the specific cases as described in the cfg file

  PSet    empty_PSet;
  VString empty_VString;

  establishDefaultEnvironment(setPrecisionDouble);
  establishModuleEnvironments(iPS, setPrecisionDouble);

  stateStack_.push(defaultState_);

  // And finally, put the state back to the way we found it originally
  //   fpuState_ = OSdefault_;
  //   fesetenv(&OSdefault_);
  fpuState_ = defaultState_;
  fesetenv(&fpuState_);
}

// EnableFloatingPointExceptions::EnableFloatingPointExceptions(EnableFloatingPointExceptions const& rhs)
// {
//    // do actual copying here;
// }

//EnableFloatingPointExceptions::~EnableFloatingPointExceptions()
//{
//}

//
// assignment operators
//
// EnableFloatingPointExceptions const& EnableFloatingPointExceptions::operator=(EnableFloatingPointExceptions const& rhs)
// {
//   //An exception safe implementation is
//   EnableFloatingPointExceptions temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

namespace {
  inline bool stateNeedsChanging(fenv_t const& current, fenv_t const& target) {
    //     bool status = current.__control_word != target.__control_word;
    //     std::cerr << "status: " << status << '\n';
    //     return status;
    return current.__control_word != target.__control_word;
  }
}


//
// member functions
//

void
EnableFloatingPointExceptions::postEndJob() {

// At EndJob, put the state of the fpu back to "OSdefault"

  fpuState_ = OSdefault_;
  fesetenv(&OSdefault_);
  if(reportSettings_) {
    edm::LogVerbatim("FPE_Enable") << "\nSettings at end job ";
    echoState();
  }
}

void 
EnableFloatingPointExceptions::preModule(ModuleDescription const& iDescription) {

// On entry to a module, find the desired state of the fpu and set it accordingly.
// Note that any module whose label does not appear in our list gets the default settings.

  String const& modName = iDescription.moduleLabel();
  std::map<String, fenv_t>::const_iterator iModule = stateMap_.find(modName);
  
  fenv_t target;
  if(iModule == stateMap_.end())  {
    target = defaultState_;
  } else {
    target = iModule->second;
  }
  if (stateNeedsChanging(fpuState_, target)) {
      fpuState_ = target;
      fesetenv(&fpuState_);
  }

  stateStack_.push(fpuState_);
  if(reportSettings_) {
    edm::LogVerbatim("FPE_Enable") << "\nSettings at begin module " << modName;
    echoState();
  }
}

void 
EnableFloatingPointExceptions::postModule(ModuleDescription const& iDescription) {

  // On exit from a module, set the state of the fpu back to what it was before entry
  stateStack_.pop();
  if (stateNeedsChanging(fpuState_, stateStack_.top())) { 
      fpuState_ = stateStack_.top();
      fesetenv(&fpuState_);
  }

  if(reportSettings_) {
    edm::LogVerbatim("FPE_Enable") << "\nSettings after end module ";
    echoState();
  }
}

void
EnableFloatingPointExceptions::controlFpe(bool divByZero, bool invalid, bool overFlow, 
					  bool underFlow, bool precisionDouble, fenv_t& result) const {
  // Local Declarations

  unsigned short int FE_PRECISION = 1<<5;
  unsigned short int suppress;

#ifdef __linux__

/*
 * NB: We are not letting users control signaling inexact (FE_INEXACT).
 */

  suppress = FE_PRECISION;
  if (!divByZero) suppress |= FE_DIVBYZERO;
  if (!invalid)   suppress |= FE_INVALID;
  if (!overFlow)  suppress |= FE_OVERFLOW;
  if (!underFlow) suppress |= FE_UNDERFLOW;
  fegetenv(&result);
  result.__control_word = suppress;
  //  fesetenv(&result);

#ifdef __i386__

  if (precisionDouble) {
    fpu_control_t cw;
    _FPU_GETCW(cw);

    cw = (cw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
    _FPU_SETCW(cw);
  }
#endif
#endif

}

void
EnableFloatingPointExceptions::echoState() {
  int femask = fegetexcept();
  edm::LogVerbatim("FPE_Enable") << "Floating point exception mask is " 
				 << std::showbase << std::hex << femask;
  
  if(femask & FE_DIVBYZERO)
    edm::LogVerbatim("FPE_Enable") << "\tDivByZero exception is on";
  else
    edm::LogVerbatim("FPE_Enable") << "\tDivByZero exception is off";
  
  if(femask & FE_INVALID)
    edm::LogVerbatim("FPE_Enable") << "\tInvalid exception is on";
  else
    edm::LogVerbatim("FPE_Enable") << "\tInvalid exception is off";
  
  if(femask & FE_OVERFLOW)
    edm::LogVerbatim("FPE_Enable") << "\tOverFlow exception is on";
  else
    edm::LogVerbatim("FPE_Enable") << "\tOverflow exception is off";
  
  if(femask & FE_UNDERFLOW)
    edm::LogVerbatim("FPE_Enable") << "\tUnderFlow exception is on";
  else
    edm::LogVerbatim("FPE_Enable") << "\tUnderFlow exception is off";
}


void EnableFloatingPointExceptions::establishDefaultEnvironment(bool setPrecisionDouble) {
  controlFpe(false, false, false, false, setPrecisionDouble, fpuState_);
  fesetenv(&fpuState_);
  if(reportSettings_) {
    edm::LogVerbatim("FPE_Enable") << "\nSettings for default";
    echoState();
  }
  defaultState_ = fpuState_;
}

// Establish an environment for each module; default is handled specially.
void EnableFloatingPointExceptions::establishModuleEnvironments(PSet const& iPS, bool setPrecisionDouble) {
  // Scan the module name list and set per-module values.  Be careful to treat
  // any user-specified default first.  If there is one, use it to override our default.
  // Then remove it from the list so we don't see it again while handling everything else.

  String const def("default");
  PSet const empty_PSet;
  VString const empty_VString;
  VString moduleNames = iPS.getUntrackedParameter<VString>("moduleNames",empty_VString);

  for (VString::const_iterator it(moduleNames.begin()), itEnd = moduleNames.end(); it != itEnd; ++it) {
    PSet secondary = iPS.getUntrackedParameter<PSet>(*it, empty_PSet);
    bool enableDivByZeroEx  = secondary.getUntrackedParameter<bool>("enableDivByZeroEx", false);
    bool enableInvalidEx    = secondary.getUntrackedParameter<bool>("enableInvalidEx",   false);
    bool enableOverFlowEx   = secondary.getUntrackedParameter<bool>("enableOverFlowEx",  false);
    bool enableUnderFlowEx  = secondary.getUntrackedParameter<bool>("enableUnderFlowEx", false);
    controlFpe(enableDivByZeroEx, enableInvalidEx, enableOverFlowEx, enableUnderFlowEx, setPrecisionDouble, fpuState_);
    fesetenv(&fpuState_);
    if(reportSettings_) {
      edm::LogVerbatim("FPE_Enable") << "\nSettings for module " << *it;
      echoState();
    }
    if (*it == def) {
      defaultState_ = fpuState_;
    } else {
      stateMap_[*it] =  fpuState_;
    }
  }
}
