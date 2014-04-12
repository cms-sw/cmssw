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

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/AllowedLabelsDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

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
fegetexcept (void) {
  fenv_t fenv;

  return fegetenv (&fenv) ? -1 : (fenv.__control & FE_ALL_EXCEPT);
}

static int
feenableexcept (unsigned int excepts) {
  fenv_t fenv;
  unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
               old_excepts;  // previous masks

  if(fegetenv (&fenv)) return -1;
  old_excepts = fenv.__control & FE_ALL_EXCEPT;

  // unmask
  fenv.__control &= ~new_excepts;
  fenv.__mxcsr   &= ~(new_excepts << 7);

  return (fesetenv (&fenv) ? -1 : old_excepts);
}

static int
fedisableexcept (unsigned int excepts) {
  fenv_t fenv;
  unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
               old_excepts;  // all previous masks

  if(fegetenv (&fenv)) return -1;
  old_excepts = fenv.__control & FE_ALL_EXCEPT;

  // mask
  fenv.__control |= new_excepts;
  fenv.__mxcsr   |= new_excepts << 7;

  return (fesetenv (&fenv) ? -1 : old_excepts);
}

#endif
#endif

namespace edm {
  namespace service {

    EnableFloatingPointExceptions::
    EnableFloatingPointExceptions(ParameterSet const& pset,
                                  ActivityRegistry & registry):
      defaultState_(0),
      stateMap_(),
      reportSettings_(false) {

      reportSettings_ = pset.getUntrackedParameter<bool>("reportSettings", false);
      bool precisionDouble = pset.getUntrackedParameter<bool>("setPrecisionDouble", true);

      if(reportSettings_)  {
        LogVerbatim("FPE_Enable") << "\nSettings in EnableFloatingPointExceptions constructor";
        echoState();
      }

      establishModuleEnvironments(pset);

      enableAndDisableExcept(defaultState_);

      setPrecision(precisionDouble);

      // Note that we must watch all of the transitions even if there are no module specific settings.
      // This is because the floating point environment may be modified by code outside of this service.
      registry.watchPostEndJob(this, &EnableFloatingPointExceptions::postEndJob);

      registry.watchPreModuleBeginStream(this, &EnableFloatingPointExceptions::preModuleBeginStream);
      registry.watchPostModuleBeginStream(this, &EnableFloatingPointExceptions::postModuleBeginStream);

      registry.watchPreModuleEndStream(this, &EnableFloatingPointExceptions::preModuleEndStream);
      registry.watchPostModuleEndStream(this, &EnableFloatingPointExceptions::postModuleEndStream);

      registry.watchPreModuleConstruction(this, &EnableFloatingPointExceptions::preModuleConstruction);
      registry.watchPostModuleConstruction(this, &EnableFloatingPointExceptions::postModuleConstruction);

      registry.watchPreModuleBeginJob(this, &EnableFloatingPointExceptions::preModuleBeginJob);
      registry.watchPostModuleBeginJob(this, &EnableFloatingPointExceptions::postModuleBeginJob);

      registry.watchPreModuleEndJob(this, &EnableFloatingPointExceptions::preModuleEndJob);
      registry.watchPostModuleEndJob(this, &EnableFloatingPointExceptions::postModuleEndJob);

      registry.watchPreModuleEvent(this, &EnableFloatingPointExceptions::preModuleEvent);
      registry.watchPostModuleEvent(this, &EnableFloatingPointExceptions::postModuleEvent);

      registry.watchPreModuleStreamBeginRun(this, &EnableFloatingPointExceptions::preModuleStreamBeginRun);
      registry.watchPostModuleStreamBeginRun(this, &EnableFloatingPointExceptions::postModuleStreamBeginRun);
      registry.watchPreModuleStreamEndRun(this, &EnableFloatingPointExceptions::preModuleStreamEndRun);
      registry.watchPostModuleStreamEndRun(this, &EnableFloatingPointExceptions::postModuleStreamEndRun);

      registry.watchPreModuleStreamBeginLumi(this, &EnableFloatingPointExceptions::preModuleStreamBeginLumi);
      registry.watchPostModuleStreamBeginLumi(this, &EnableFloatingPointExceptions::postModuleStreamBeginLumi);
      registry.watchPreModuleStreamEndLumi(this, &EnableFloatingPointExceptions::preModuleStreamEndLumi);
      registry.watchPostModuleStreamEndLumi(this, &EnableFloatingPointExceptions::postModuleStreamEndLumi);

      registry.watchPreModuleGlobalBeginRun(this, &EnableFloatingPointExceptions::preModuleGlobalBeginRun);
      registry.watchPostModuleGlobalBeginRun(this, &EnableFloatingPointExceptions::postModuleGlobalBeginRun);
      registry.watchPreModuleGlobalEndRun(this, &EnableFloatingPointExceptions::preModuleGlobalEndRun);
      registry.watchPostModuleGlobalEndRun(this, &EnableFloatingPointExceptions::postModuleGlobalEndRun);

      registry.watchPreModuleGlobalBeginLumi(this, &EnableFloatingPointExceptions::preModuleGlobalBeginLumi);
      registry.watchPostModuleGlobalBeginLumi(this, &EnableFloatingPointExceptions::postModuleGlobalBeginLumi);
      registry.watchPreModuleGlobalEndLumi(this, &EnableFloatingPointExceptions::preModuleGlobalEndLumi);
      registry.watchPostModuleGlobalEndLumi(this, &EnableFloatingPointExceptions::postModuleGlobalEndLumi);
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

      for(VString::const_iterator it(moduleNames.begin()), itEnd = moduleNames.end(); it != itEnd; ++it) {
        ParameterSet const& modulePSet = pset.getUntrackedParameterSet(*it, empty_PSet);
        bool enableDivByZeroEx  = modulePSet.getUntrackedParameter<bool>("enableDivByZeroEx", false);
        bool enableInvalidEx    = modulePSet.getUntrackedParameter<bool>("enableInvalidEx",   false);
        bool enableOverFlowEx   = modulePSet.getUntrackedParameter<bool>("enableOverFlowEx",  false);
        bool enableUnderFlowEx  = modulePSet.getUntrackedParameter<bool>("enableUnderFlowEx", false);

        fpu_flags_type flags = 0;
        if(enableDivByZeroEx) flags |= FE_DIVBYZERO;
        if(enableInvalidEx)   flags |= FE_INVALID;
        if(enableOverFlowEx)  flags |= FE_OVERFLOW;
        if(enableUnderFlowEx) flags |= FE_UNDERFLOW;
        enableAndDisableExcept(flags);

        fpu_flags_type fpuState = fegetexcept();
        assert(flags == fpuState);

        if(reportSettings_) {
          LogVerbatim("FPE_Enable") << "\nSettings for module " << *it;
          echoState();
        }
        if(*it == def) {
          defaultState_ = fpuState;
        }
        else {
          stateMap_[*it] =  fpuState;
        }
      }
    }

    void
    EnableFloatingPointExceptions::postEndJob() {

      if(reportSettings_) {
        LogVerbatim("FPE_Enable") << "\nSettings after endJob ";
        echoState();
      }
    }

    void
    EnableFloatingPointExceptions::
    preModuleConstruction(ModuleDescription const& md) {
      preActions(md, "construction");
    }

    void
    EnableFloatingPointExceptions::
    postModuleConstruction(ModuleDescription const& md) {
      postActions(md, "construction");
    }

    void 
    EnableFloatingPointExceptions::
    preModuleBeginJob(ModuleDescription const& md) {
      preActions(md, "beginJob");
    }

    void 
    EnableFloatingPointExceptions::
    postModuleBeginJob(ModuleDescription const& md) {
      postActions(md, "beginJob");
    }

    void
    EnableFloatingPointExceptions::
    preModuleEndJob(ModuleDescription const& md) {
      preActions(md, "endJob");
    }

    void
    EnableFloatingPointExceptions::
    postModuleEndJob(ModuleDescription const& md) {
      postActions(md, "endJob");
    }

    void
    EnableFloatingPointExceptions::
    preModuleBeginStream(StreamContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "beginStream");
    }

    void
    EnableFloatingPointExceptions::
    postModuleBeginStream(StreamContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "beginStream");
    }

    void
    EnableFloatingPointExceptions::
    preModuleEndStream(StreamContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "endStream");
    }

    void
    EnableFloatingPointExceptions::
    postModuleEndStream(StreamContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "endStream");
    }

    void
    EnableFloatingPointExceptions::
    preModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "globalBeginRun");
    }

    void
    EnableFloatingPointExceptions::
    postModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "globalBeginRun");
    }

    void
    EnableFloatingPointExceptions::
    preModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "globalEndRun");
    }

    void
    EnableFloatingPointExceptions::
    postModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "globalEndRun");
    }

    void
    EnableFloatingPointExceptions::
    preModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "globalBeginLumi");
    }

    void
    EnableFloatingPointExceptions::
    postModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "globalBeginLumi");
    }

    void
    EnableFloatingPointExceptions::
    preModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "globalEndLumi");
    }

    void
    EnableFloatingPointExceptions::
    postModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "globalEndLumi");
    }

    void
    EnableFloatingPointExceptions::
    preModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "streamBeginRun");
    }

    void
    EnableFloatingPointExceptions::
    postModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "streamBeginRun");
    }

    void
    EnableFloatingPointExceptions::
    preModuleStreamEndRun(StreamContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "streamEndRun");
    }

    void
    EnableFloatingPointExceptions::
    postModuleStreamEndRun(StreamContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "streamEndRun");
    }

    void
    EnableFloatingPointExceptions::
    preModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "streamBeginLumi");
    }

    void
    EnableFloatingPointExceptions::
    postModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "streamBeginLumi");
    }

    void
    EnableFloatingPointExceptions::
    preModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "streamEndLumi");
    }

    void
    EnableFloatingPointExceptions::
    postModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "streamEndLumi");
    }

    void
    EnableFloatingPointExceptions::
    preModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
      preActions(mcc, "event");
    }

    void
    EnableFloatingPointExceptions::
    postModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
      postActions(mcc, "event");
    }
    
    void
    EnableFloatingPointExceptions::
    fillDescriptions(ConfigurationDescriptions & descriptions) {
      ParameterSetDescription desc;

      desc.addUntracked<bool>("reportSettings", false)->setComment(
        "Log FPE settings at different phases of the job.");
      desc.addUntracked<bool>("setPrecisionDouble", true)->setComment(
        "Set the FPU to use double precision");

      ParameterSetDescription validator;
      validator.setComment("FPU exceptions to enable/disable for the requested module");
      validator.addUntracked<bool>("enableDivByZeroEx", false)->setComment(
        "Enable/disable exception for 'divide by zero'");
      validator.addUntracked<bool>("enableInvalidEx",   false)->setComment(
        "Enable/disable exception for 'invalid' math operations (e.g. sqrt(-1))");
      validator.addUntracked<bool>("enableOverFlowEx",  false)->setComment(
        "Enable/disable exception for numeric 'overflow' (value to big for type)");
      validator.addUntracked<bool>("enableUnderFlowEx", false)->setComment(
        "Enable/disable exception for numeric 'underflow' (value to small to be represented accurately)");

      AllowedLabelsDescription<ParameterSetDescription> node("moduleNames", validator, false);
      node.setComment("Contains the names for PSets where the PSet name matches the label of a module for which you want to modify the FPE");
      desc.addNode(node);

      descriptions.add("EnableFloatingPointExceptions", desc);
      descriptions.setComment("This service allows you to control the FPU and its exceptions on a per module basis.");
    }


    void 
    EnableFloatingPointExceptions::
    preActions(ModuleDescription const& description,
               char const* debugInfo) {

      std::string const& moduleLabel = description.moduleLabel();
      std::map<std::string, fpu_flags_type>::const_iterator iModule = stateMap_.find(moduleLabel);

      fpu_flags_type fpuState = defaultState_;

      if(iModule != stateMap_.end())  {
        fpuState = iModule->second;
      }
      enableAndDisableExcept(fpuState);

      if(reportSettings_) {
        LogVerbatim("FPE_Enable")
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

      enableAndDisableExcept(defaultState_);

      if(reportSettings_) {
        LogVerbatim("FPE_Enable")
          << "\nSettings for module label \""
          << description.moduleLabel()
          << "\" after "
          << debugInfo;
        echoState();
      }
    }

    void 
    EnableFloatingPointExceptions::
    preActions(ModuleCallingContext const& mcc,
               char const* debugInfo) {

      std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
      std::map<std::string, fpu_flags_type>::const_iterator iModule = stateMap_.find(moduleLabel);

      fpu_flags_type fpuState = defaultState_;

      if(iModule != stateMap_.end())  {
        fpuState = iModule->second;
      }
      enableAndDisableExcept(fpuState);

      if(reportSettings_) {
        LogVerbatim("FPE_Enable")
          << "\nSettings for module label \""
          << moduleLabel
          << "\" before "
          << debugInfo;
        echoState();
      }
    }

    void 
    EnableFloatingPointExceptions::
    postActions(ModuleCallingContext const& mcc, char const* debugInfo) {

      fpu_flags_type fpuState = defaultState_;

      edm::ModuleCallingContext const* previous_mcc = mcc.previousModuleOnThread();
      if(previous_mcc) {
        std::map<std::string, fpu_flags_type>::const_iterator iModule = stateMap_.find(previous_mcc->moduleDescription()->moduleLabel());
        if(iModule != stateMap_.end())  {
          fpuState = iModule->second;
        }
      }
      enableAndDisableExcept(fpuState);

      if(reportSettings_) {
        LogVerbatim("FPE_Enable")
          << "\nSettings for module label \""
          << mcc.moduleDescription()->moduleLabel()
          << "\" after "
          << debugInfo;
        echoState();
      }
    }

#ifdef __linux__
#ifdef __i386__
    // Note that __i386__ flag is not set on x86_64 architectures.
    // As far as I know we use the empty version of the setPrecision
    // function on all architectures CMS currently supports.
    // Here is my understanding of this from the articles I found with
    // google. I should warn that none of those articles were directly
    // from the compiler writers or Intel or any authoritative
    // source and I am not really an expert on this subject. We use the
    // setPrecision function to force the math processor to perform floating
    // point calculations internally with 64 bits of precision and not use
    // 80 bit extended precision internally for calculations. 80 bit extended
    // precision is used with the x87 instruction set which is the default
    // for most 32 bit Intel architectures. When it is used there are problems
    // of different sorts such as nonreproducible results, mostly due
    // to rounding issues. This was important before CMS switched from
    // 32 bit to 64 bit architectures. On 64 bit x86_64 platforms
    // SSE instructions are used instead of x87 instructions.
    // The whole issue is obsolete as SSE instructions do not ever
    // use extended 80 bit precision in floating point calculations. Although
    // new CPUs still support the x87 instruction set for floating point
    // calculations for various reasons (mostly backward compatibility I
    // think), most compilers write SSE instructions only. It might be
    // that compiler flags can be set to force use of x87 instructions, but
    // as far as I know we do not do that for CMS.
    void
    EnableFloatingPointExceptions::setPrecision(bool precisionDouble) {
      if(precisionDouble) {
        fpu_control_t cw;
        _FPU_GETCW(cw);

        cw = (cw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
        _FPU_SETCW(cw);
      }
    }
#else
    void
    EnableFloatingPointExceptions::setPrecision(bool /*precisionDouble*/) {
    }
#endif
#else
    void
    EnableFloatingPointExceptions::setPrecision(bool /*precisionDouble*/) {
    }
#endif

    void
    EnableFloatingPointExceptions::enableAndDisableExcept(fpu_flags_type target) {
      feclearexcept(FE_ALL_EXCEPT);
      fpu_flags_type current = fegetexcept();
      fpu_flags_type exceptionsToModify = current ^ target;
      fpu_flags_type exceptionsToEnable = 0;
      fpu_flags_type exceptionsToDisable = 0;

      if(exceptionsToModify & FE_DIVBYZERO) {
        if(target & FE_DIVBYZERO) {
          exceptionsToEnable |= FE_DIVBYZERO;
        }
        else {
          exceptionsToDisable |= FE_DIVBYZERO;
        }
      }
      if(exceptionsToModify & FE_INVALID) {
        if(target & FE_INVALID) {
          exceptionsToEnable |= FE_INVALID;
        }
        else {
          exceptionsToDisable |= FE_INVALID;
        }
      }
      if(exceptionsToModify & FE_OVERFLOW) {
        if(target & FE_OVERFLOW) {
          exceptionsToEnable |= FE_OVERFLOW;
        }
        else {
          exceptionsToDisable |= FE_OVERFLOW;
        }
      }
      if(exceptionsToModify & FE_UNDERFLOW) {
        if(target & FE_UNDERFLOW) {
          exceptionsToEnable |= FE_UNDERFLOW;
        }
        else {
          exceptionsToDisable |= FE_UNDERFLOW;
        }
      }
      if(exceptionsToEnable != 0) {
        feenableexcept(exceptionsToEnable);
      }
      if(exceptionsToDisable != 0) {
        fedisableexcept(exceptionsToDisable);
      }
    }

    void
    EnableFloatingPointExceptions::echoState() const {
      feclearexcept(FE_ALL_EXCEPT);
      fpu_flags_type femask = fegetexcept();
      LogVerbatim("FPE_Enable") << "Floating point exception mask is " 
				 << std::showbase << std::hex << femask;
 
      if(femask & FE_DIVBYZERO)
        LogVerbatim("FPE_Enable") << "\tDivByZero exception is on";
      else
        LogVerbatim("FPE_Enable") << "\tDivByZero exception is off";
  
      if(femask & FE_INVALID)
        LogVerbatim("FPE_Enable") << "\tInvalid exception is on";
      else
        LogVerbatim("FPE_Enable") << "\tInvalid exception is off";
 
      if(femask & FE_OVERFLOW)
        LogVerbatim("FPE_Enable") << "\tOverFlow exception is on";
      else
        LogVerbatim("FPE_Enable") << "\tOverflow exception is off";
  
      if(femask & FE_UNDERFLOW)
        LogVerbatim("FPE_Enable") << "\tUnderFlow exception is on";
      else
        LogVerbatim("FPE_Enable") << "\tUnderFlow exception is off";
    }

  } // namespace edm
} // namespace service
