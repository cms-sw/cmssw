#ifndef FWCore_Services_FpeHandler_h
#define FWCore_Services_FpeHandler_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     EnableFloatingPointExceptions
// 
/**\class EnableFloatingPointExceptions EnableFloatingPointExceptions.h FWCore/Services/src/EnableFloatingPointExceptions.h

    Description: This service gives cmsRun users the ability to configure the 
    behavior of the Floating Point (FP) environment.  There are two separate
    aspects of the FP environment this service can control: 

        1. floating-point exceptions (on a module by module basis if desired)
        2. precision control on x87 FP processors.

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

    One can also control the precision of floating point operations in x87 FP
    processors as follows:

      process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions",
          setPrecisionDouble = cms.untracked.bool(True)
      )

    If set true (the default if the service is used), the floating precision in
    the x87 math processor will be set to round results of addition,
    subtraction, multiplication, division, and square root to 64 bits after
    each operation instead of the x87 default, which is 80 bits for values in
    registers (this is the default you get if this service is not used at all).

    The precision control only affects Intel and AMD 32 bit CPUs under LINUX.
    We have not implemented precision control in the service for other CPUs yet
    (some other CPUs round to 64 bits by default and some other CPUs do not allow
    control of the precision of floating point calculations, the behavior of
    other CPUs may need more study in the future).
*/
//
// Original Author:  E. Sexton-Kennedy
//         Created:  Tue Apr 11 13:43:16 CDT 2006
//

#include <string>
#include <map>
#include <stack>

namespace edm {

   class ParameterSet;
   struct ActivityRegistry;
   class ModuleDescription;
   class ConfigurationDescriptions;

   namespace service {

      class EnableFloatingPointExceptions {
      public:
         typedef int fpu_flags_type;
         EnableFloatingPointExceptions(ParameterSet const& pset,
                                       ActivityRegistry & registry);

	 void postEndJob();

         void preModuleBeginJob(ModuleDescription const& description);
         void postModuleBeginJob(ModuleDescription const& description);
         void preModuleEndJob(ModuleDescription const& description);
         void postModuleEndJob(ModuleDescription const& description);

         void preModuleBeginRun(ModuleDescription const& description);
         void postModuleBeginRun(ModuleDescription const& description);
         void preModuleEndRun(ModuleDescription const& description);
         void postModuleEndRun(ModuleDescription const& description);

         void preModuleBeginLumi(ModuleDescription const& description);
         void postModuleBeginLumi(ModuleDescription const& description);
         void preModuleEndLumi(ModuleDescription const& description);
         void postModuleEndLumi(ModuleDescription const& description);

         void preModule(ModuleDescription const& description);
         void postModule(ModuleDescription const& description);

         static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      private:
         typedef std::string String;
         typedef ParameterSet PSet;

         void preActions(ModuleDescription const& description,
                         char const* debugInfo);

         void postActions(ModuleDescription const& description,
                          char const* debugInfo);

         void controlFpe(bool divByZero, bool invalid, bool overFlow,
                         bool underFlow, bool precisionDouble) const;

         void echoState() const;
         void establishDefaultEnvironment(bool precisionDouble);
         void establishModuleEnvironments(PSet const& pset, bool precisionDouble);


         fpu_flags_type fpuState_;
         fpu_flags_type defaultState_;
         fpu_flags_type OSdefault_;
         std::map<String, fpu_flags_type> stateMap_;
         std::stack<fpu_flags_type> stateStack_;
         bool reportSettings_;
      };
   }
}
#endif
