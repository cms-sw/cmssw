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

#include "FWCore/Services/src/EnableFloatingPointExceptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/AllowedLabelsDescription.h"

#include <cassert>
#include <fenv.h>
#include <vector>
#ifdef __linux__
#include <fpu_control.h>
#else
#ifdef __APPLE__
//TAKEN FROM
// http://public.kitware.com/Bug/file_download.php?file_id=3215&type=bug
static int
fegetexcept (void)
{
  static fenv_t fenv;

  return fegetenv (&fenv) ? -1 : (fenv.__control & FE_ALL_EXCEPT);
}

static int
feenableexcept (unsigned int excepts)
{
  static fenv_t fenv;
  unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
               old_excepts;  // previous masks

  if ( fegetenv (&fenv) ) return -1;
  old_excepts = fenv.__control & FE_ALL_EXCEPT;

  // unmask
  fenv.__control &= ~new_excepts;
  fenv.__mxcsr   &= ~(new_excepts << 7);

  return ( fesetenv (&fenv) ? -1 : old_excepts );
}

static int
fedisableexcept (unsigned int excepts)
{
  static fenv_t fenv;
  unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
               old_excepts;  // all previous masks

  if ( fegetenv (&fenv) ) return -1;
  old_excepts = fenv.__control & FE_ALL_EXCEPT;

  // mask
  fenv.__control |= new_excepts;
  fenv.__mxcsr   |= new_excepts << 7;

  return ( fesetenv (&fenv) ? -1 : old_excepts );
}

#endif
#endif

namespace edm {
  namespace service {

    EnableFloatingPointExceptions::
    EnableFloatingPointExceptions(ParameterSet const& pset,
                                  ActivityRegistry & registry):
      fpuState_(0),
      defaultState_(0),
      stateMap_(),
      stateStack_(),
      reportSettings_(false) {

      reportSettings_ = pset.getUntrackedParameter<bool>("reportSettings", false);
      bool precisionDouble = pset.getUntrackedParameter<bool>("setPrecisionDouble", true);

      if (reportSettings_)  {
        edm::LogVerbatim("FPE_Enable") << "\nSettings in EnableFloatingPointExceptions constructor";
        echoState();
      }

      establishModuleEnvironments(pset);

      stateStack_.push(defaultState_);
      fpuState_ = defaultState_;
      enableAndDisableExcept(defaultState_);

      setPrecision(precisionDouble);

      // Note that we must watch all of the transitions even if there are no module specific settings.
      // This is because the floating point environment may be modified by code outside of this service.
      registry.watchPostEndJob(this,&EnableFloatingPointExceptions::postEndJob);

      registry.watchPreModuleBeginJob(this, &EnableFloatingPointExceptions::preModuleBeginJob);
      registry.watchPostModuleBeginJob(this, &EnableFloatingPointExceptions::postModuleBeginJob);
      registry.watchPreModuleEndJob(this, &EnableFloatingPointExceptions::preModuleEndJob);
      registry.watchPostModuleEndJob(this, &EnableFloatingPointExceptions::postModuleEndJob);

      registry.watchPreModuleBeginRun(this, &EnableFloatingPointExceptions::preModuleBeginRun);
      registry.watchPostModuleBeginRun(this, &EnableFloatingPointExceptions::postModuleBeginRun);
      registry.watchPreModuleEndRun(this, &EnableFloatingPointExceptions::preModuleEndRun);
      registry.watchPostModuleEndRun(this, &EnableFloatingPointExceptions::postModuleEndRun);

      registry.watchPreModuleBeginLumi(this, &EnableFloatingPointExceptions::preModuleBeginLumi);
      registry.watchPostModuleBeginLumi(this, &EnableFloatingPointExceptions::postModuleBeginLumi);
      registry.watchPreModuleEndLumi(this, &EnableFloatingPointExceptions::preModuleEndLumi);
      registry.watchPostModuleEndLumi(this, &EnableFloatingPointExceptions::postModuleEndLumi);

      registry.watchPreModule(this, &EnableFloatingPointExceptions::preModule);
      registry.watchPostModule(this, &EnableFloatingPointExceptions::postModule);
    }

    // Establish an environment for each module; default is handled specially.
    void
    EnableFloatingPointExceptions::establishModuleEnvironments(ParameterSet const& pset) {

      // Scan the module name list and set per-module values.  Be careful to treat
      // any user-specified default first.  If there is one, use it to override our default.
      // Then remove it from the list so we don't see it again while handling everything else.

      typedef std::vector<std::string> VString;

      std::string const def("default");
      ParameterSet const empty_PSet;
      VString const empty_VString;
      VString moduleNames = pset.getUntrackedParameter<VString>("moduleNames", empty_VString);

      for (VString::const_iterator it(moduleNames.begin()), itEnd = moduleNames.end(); it != itEnd; ++it) {
        ParameterSet const& modulePSet = pset.getUntrackedParameterSet(*it, empty_PSet);
        bool enableDivByZeroEx  = modulePSet.getUntrackedParameter<bool>("enableDivByZeroEx", false);
        bool enableInvalidEx    = modulePSet.getUntrackedParameter<bool>("enableInvalidEx",   false);
        bool enableOverFlowEx   = modulePSet.getUntrackedParameter<bool>("enableOverFlowEx",  false);
        bool enableUnderFlowEx  = modulePSet.getUntrackedParameter<bool>("enableUnderFlowEx", false);

        fpu_flags_type flags = 0;
        if (enableDivByZeroEx) flags |= FE_DIVBYZERO;
        if (enableInvalidEx)   flags |= FE_INVALID;
        if (enableOverFlowEx)  flags |= FE_OVERFLOW;
        if (enableUnderFlowEx) flags |= FE_UNDERFLOW;
        enableAndDisableExcept(flags);

        fpuState_ = fegetexcept();
        assert(flags == fpuState_);

        if (reportSettings_) {
          edm::LogVerbatim("FPE_Enable") << "\nSettings for module " << *it;
          echoState();
        }
        if (*it == def) {
          defaultState_ = fpuState_;
        }
        else {
          stateMap_[*it] =  fpuState_;
        }
      }
    }

    void
    EnableFloatingPointExceptions::postEndJob() {

      if (reportSettings_) {
        edm::LogVerbatim("FPE_Enable") << "\nSettings after endJob ";
        echoState();
      }
    }

    void 
    EnableFloatingPointExceptions::
    preModuleBeginJob(ModuleDescription const& description) {
      preActions(description, "beginJob");
    }

    void 
    EnableFloatingPointExceptions::
    postModuleBeginJob(ModuleDescription const& description) {
      postActions(description, "beginJob");
    }

    void 
    EnableFloatingPointExceptions::
    preModuleEndJob(ModuleDescription const& description) {
      preActions(description, "endJob");
    }

    void 
    EnableFloatingPointExceptions::
    postModuleEndJob(ModuleDescription const& description) {
      postActions(description, "endJob");
    }

    void 
    EnableFloatingPointExceptions::
    preModuleBeginRun(ModuleDescription const& description) {
      preActions(description, "beginRun");
    }

    void 
    EnableFloatingPointExceptions::
    postModuleBeginRun(ModuleDescription const& description) {
      postActions(description, "beginRun");
    }

    void 
    EnableFloatingPointExceptions::
    preModuleEndRun(ModuleDescription const& description) {
      preActions(description, "endRun");
    }

    void 
    EnableFloatingPointExceptions::
    postModuleEndRun(ModuleDescription const& description) {
      postActions(description, "endRun");
    }

    void
    EnableFloatingPointExceptions::
    preModuleBeginLumi(ModuleDescription const& description) {
      preActions(description, "beginLumi");
    }

    void 
    EnableFloatingPointExceptions::
    postModuleBeginLumi(ModuleDescription const& description) {
      postActions(description, "beginLumi");
    }

    void 
    EnableFloatingPointExceptions::
    preModuleEndLumi(ModuleDescription const& description) {
      preActions(description, "endLumi");
    }

    void 
    EnableFloatingPointExceptions::
    postModuleEndLumi(ModuleDescription const& description) {
      postActions(description, "endLumi");
    }

    void 
    EnableFloatingPointExceptions::
    preModule(ModuleDescription const& description) {
      preActions(description, "event");
    }

    void 
    EnableFloatingPointExceptions::
    postModule(ModuleDescription const& description) {
      postActions(description, "event");
    }

    void
    EnableFloatingPointExceptions::
    fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
      edm::ParameterSetDescription desc;

      desc.addUntracked<bool>("reportSettings", false)->setComment(
        "Log FPE settings at different phases of the job."
                                                               );
      desc.addUntracked<bool>("setPrecisionDouble", true)->setComment(
        "Set the FPU to use double precision");

      edm::ParameterSetDescription validator;
      validator.setComment("FPU exceptions to enable/disable for the requested module");
      validator.addUntracked<bool>("enableDivByZeroEx", false)->setComment(
        "Enable/disable exception for 'divide by zero'");
      validator.addUntracked<bool>("enableInvalidEx",   false)->setComment(
        "Enable/disable exception for 'invalid' math operations (e.g. sqrt(-1))");
      validator.addUntracked<bool>("enableOverFlowEx",  false)->setComment(
        "Enable/disable exception for numeric 'overflow' (value to big for type)");
      validator.addUntracked<bool>("enableUnderFlowEx", false)->setComment(
        "Enable/disable exception for numeric 'underflow' (value to small to be represented accurately)");

      edm::AllowedLabelsDescription<edm::ParameterSetDescription> node("moduleNames", validator, false);
      node.setComment("Contains the names for PSets where the PSet name matches the label of a module for which you want to modify the FPE");
      desc.addNode(node);

      descriptions.add("EnableFloatingPointExceptions", desc);
      descriptions.setComment("This service allows you to control the FPU and its exceptions on a per module basis.");
    }


    void 
    EnableFloatingPointExceptions::
    preActions(ModuleDescription const& description,
               char const* debugInfo) {

      // On entry to a module, find the desired state of the fpu and set it
      // accordingly. Note that any module whose label does not appear in
      // our list gets the default settings.

      std::string const& moduleLabel = description.moduleLabel();
      std::map<std::string, fpu_flags_type>::const_iterator iModule = stateMap_.find(moduleLabel);

      if (iModule == stateMap_.end())  {
        fpuState_ = defaultState_;
      }
      else {
        fpuState_ = iModule->second;
      }
      enableAndDisableExcept(fpuState_);
      stateStack_.push(fpuState_);

      if (reportSettings_) {
        edm::LogVerbatim("FPE_Enable")
          << "\nSettings for module label \""
          << moduleLabel
          << "\" before "
          << debugInfo;
        echoState();
      }
    }

    void 
    EnableFloatingPointExceptions::
    postActions(ModuleDescription const& description, char const* debugInfo) {
      // On exit from a module, set the state of the fpu back to what
      // it was before entry
      stateStack_.pop();
      fpuState_ = stateStack_.top();
      enableAndDisableExcept(fpuState_);

      if (reportSettings_) {
        edm::LogVerbatim("FPE_Enable")
          << "\nSettings for module label \""
          << description.moduleLabel()
          << "\" after "
          << debugInfo;
        echoState();
      }
    }

    void
    EnableFloatingPointExceptions::setPrecision(bool precisionDouble) {
#ifdef __linux__
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
    EnableFloatingPointExceptions::enableAndDisableExcept(fpu_flags_type target) {
      feclearexcept(FE_ALL_EXCEPT);
      fpu_flags_type current = fegetexcept();
      fpu_flags_type exceptionsToModify = current ^ target;
      fpu_flags_type exceptionsToEnable = 0;
      fpu_flags_type exceptionsToDisable = 0;

      if (exceptionsToModify & FE_DIVBYZERO) {
        if (target & FE_DIVBYZERO) {
          exceptionsToEnable |= FE_DIVBYZERO;
        }
        else {
          exceptionsToDisable |= FE_DIVBYZERO;
        }
      }
      if (exceptionsToModify & FE_INVALID) {
        if (target & FE_INVALID) {
          exceptionsToEnable |= FE_INVALID;
        }
        else {
          exceptionsToDisable |= FE_INVALID;
        }
      }
      if (exceptionsToModify & FE_OVERFLOW) {
        if (target & FE_OVERFLOW) {
          exceptionsToEnable |= FE_OVERFLOW;
        }
        else {
          exceptionsToDisable |= FE_OVERFLOW;
        }
      }
      if (exceptionsToModify & FE_UNDERFLOW) {
        if (target & FE_UNDERFLOW) {
          exceptionsToEnable |= FE_UNDERFLOW;
        }
        else {
          exceptionsToDisable |= FE_UNDERFLOW;
        }
      }
      if (exceptionsToEnable != 0) {
        feenableexcept(exceptionsToEnable);
      }
      if (exceptionsToDisable != 0) {
        fedisableexcept(exceptionsToDisable);
      }
    }

    void
    EnableFloatingPointExceptions::echoState() const {
      feclearexcept(FE_ALL_EXCEPT);
      fpu_flags_type femask = fegetexcept();
      edm::LogVerbatim("FPE_Enable") << "Floating point exception mask is " 
				 << std::showbase << std::hex << femask;
 
      if (femask & FE_DIVBYZERO)
        edm::LogVerbatim("FPE_Enable") << "\tDivByZero exception is on";
      else
        edm::LogVerbatim("FPE_Enable") << "\tDivByZero exception is off";
  
      if (femask & FE_INVALID)
        edm::LogVerbatim("FPE_Enable") << "\tInvalid exception is on";
      else
        edm::LogVerbatim("FPE_Enable") << "\tInvalid exception is off";
 
      if (femask & FE_OVERFLOW)
        edm::LogVerbatim("FPE_Enable") << "\tOverFlow exception is on";
      else
        edm::LogVerbatim("FPE_Enable") << "\tOverflow exception is off";
  
      if (femask & FE_UNDERFLOW)
        edm::LogVerbatim("FPE_Enable") << "\tUnderFlow exception is on";
      else
        edm::LogVerbatim("FPE_Enable") << "\tUnderFlow exception is off";
    }

  } // namespace edm
} // namespace service
