#ifndef FWCore_Services_FpeHandler_h
#define FWCore_Services_FpeHandler_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     EnableFloatingPointExceptions
// 
/** \class edm::service::EnableFloatingPointExceptions

    Description: This service gives cmsRun users the ability to configure the 
    behavior of the Floating Point (FP) environment.  There are two separate
    aspects of the FP environment this service can control: 

        1. floating-point exceptions (on a module by module basis if desired)
        2. precision control on x87 FP processors (obsolete for 64 bit x86_64)

    If you do not use the service at all, floating point exceptions will not
    be trapped anywhere (FP exceptions will not cause a crash).  Add something
    like the following to the configuration file to enable to the exceptions:

      process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions",
          moduleNames = cms.untracked.vstring(
              'default',
              'sendMessages1',
              'sendMessages2'
          ),
          default = cms.untracked.PSet(
              enableOverFlowEx = cms.untracked.bool(False),
              enableDivByZeroEx = cms.untracked.bool(False),
              enableInvalidEx = cms.untracked.bool(False),
              enableUnderFlowEx = cms.untracked.bool(False)
          ),
          sendMessages1 = cms.untracked.PSet(
              enableOverFlowEx = cms.untracked.bool(False),
              enableDivByZeroEx = cms.untracked.bool(True),
              enableInvalidEx = cms.untracked.bool(False),
              enableUnderFlowEx = cms.untracked.bool(False)
          ),
          sendMessages2 = cms.untracked.PSet(
              enableOverFlowEx = cms.untracked.bool(False),
              enableDivByZeroEx = cms.untracked.bool(False),
              enableInvalidEx = cms.untracked.bool(True),
              enableUnderFlowEx = cms.untracked.bool(False)
          )
      )

    In this example, the "Divide By Zero" exception is enabled only for the
    module with label "sendMessages1", the "Invalid" exception is enabled
    only for the module with label sendMessages2 and no floating point
    exceptions are otherwise enabled.

    The defaults for these options are currently all false.  (in an earlier
    version DivByZero, Invalid, and Overflow defaulted to true, we hope to
    return to those defaults someday when the frequency of such exceptions
    has decreased)

    Enabling exceptions is very useful if you are trying to track down where a
    floating point value of 'nan' or 'inf' is being generated and is even better
    if the goal is to eliminate them.

    Warning. The flags that control the behavior of floating point are globally
    available. Anything could potentially change them at any point in time, including
    the module itself and third party code that module calls. In a threading
    environment if the module waits in the middle of execution and others tasks
    run on the same thread, they could potentially change the behavior.
    Also if the module internally creates tasks that run on other threads
    those will run with whatever settings are on that thread unless special code
    has been written in the module to set the floating point exception control
    flags.

    Note that we did tests to determine that the floating point exception control
    of feenableexcept was thread local and therefore could be use with the
    multithreaded framework. Although the functions in fenv.h that are part
    of the standard are required to be thread local, we were unable to find
    any documentation about this for feenableexcept which is an extension
    of the GCC compiler. Hopefully our test on one machine can be safely
    extrapolated to all machines ...

    Precision control is obsolete in x86_64 architectures and other architectures
    CMS currently uses (maybe we should remove this part of the service). It has
    no effect. The x86_64 architecture uses SSE for floating point calculations,
    not x87. SSE always uses 64 bit precision.  Precision control  works for x87
    floating point calculations as follows:

      process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions",
          setPrecisionDouble = cms.untracked.bool(True)
      )

    If set true (the default if the service is used), the floating precision in
    the x87 math processor will be set to round results of addition,
    subtraction, multiplication, division, and square root to 64 bits after
    each operation instead of the x87 default, which is 80 bits for values in
    registers (this is the default you get if this service is not used at all).

    The precision control only affects Intel and AMD 32 bit CPUs under LINUX.
    We have not implemented precision control in the service for other CPUs.
    (most other CPUs round to 64 bits by default and/or do not allow
    control of the precision of floating point calculations).
*/
//
// Original Author:  E. Sexton-Kennedy
//         Created:  Tue Apr 11 13:43:16 CDT 2006
//

#include <string>
#include <map>

namespace edm {

   class ParameterSet;
   class ActivityRegistry;
   class ModuleDescription;
   class StreamContext;
   class ModuleCallingContext;
   class GlobalContext;
   class ConfigurationDescriptions;

   namespace service {

      class EnableFloatingPointExceptions {
      public:

         typedef int fpu_flags_type;

         EnableFloatingPointExceptions(ParameterSet const& pset,
                                       ActivityRegistry & registry);
         void postEndJob();

         void preModuleConstruction(ModuleDescription const&);
         void postModuleConstruction(ModuleDescription const&);

         void preModuleBeginJob(ModuleDescription const&);
         void postModuleBeginJob(ModuleDescription const&);

         void preModuleEndJob(ModuleDescription const& md);
         void postModuleEndJob(ModuleDescription const& md);

         void preModuleBeginStream(StreamContext const&, ModuleCallingContext const&);
         void postModuleBeginStream(StreamContext const&, ModuleCallingContext const&);

         void preModuleEndStream(StreamContext const&, ModuleCallingContext const&);
         void postModuleEndStream(StreamContext const&, ModuleCallingContext const&);

         void preModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
         void postModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
         void preModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);
         void postModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);

         void preModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
         void postModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
         void preModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);
         void postModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);

         void preModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
         void postModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
         void preModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);
         void postModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);

         void preModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
         void postModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
         void preModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);
         void postModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);

         void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
         void postModuleEvent(StreamContext const&, ModuleCallingContext const&);
         
         static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      private:

         void establishModuleEnvironments(ParameterSet const&);

         void preActions(ModuleDescription const&,
                         char const* debugInfo);

         void postActions(ModuleDescription const&,
                          char const* debugInfo);

         void preActions(ModuleCallingContext const&,
                         char const* debugInfo);

         void postActions(ModuleCallingContext const&,
                          char const* debugInfo);

         void setPrecision(bool precisionDouble);

         void enableAndDisableExcept(fpu_flags_type target);

         void echoState() const;

         fpu_flags_type defaultState_;
         std::map<std::string, fpu_flags_type> stateMap_;
         bool reportSettings_;
      };
   }
}
#endif
