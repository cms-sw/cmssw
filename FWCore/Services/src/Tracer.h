#ifndef FWCore_Services_Tracer_h
#define FWCore_Services_Tracer_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     Tracer
// 
/**\class Tracer Tracer.h FWCore/Services/interface/Tracer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Sep  8 14:35:45 EDT 2005
//

// system include files

// user include files

// forward declarations

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <string>

namespace edm {
   class ConfigurationDescriptions;
   class GlobalContext;
   class HLTPathStatus;
   class LuminosityBlock;
   class ModuleCallingContext;
   class ModuleDescription;
   class PathContext;
   class Run;
   class StreamContext;

   namespace service {
      class Tracer {
public:
         Tracer(const ParameterSet&,ActivityRegistry&);
         
         static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

         void postBeginJob();
         void postEndJob();
         
         void preSource();
         void postSource();

         void preSourceLumi();
         void postSourceLumi();

         void preSourceRun();
         void postSourceRun();

         void preOpenFile(std::string const&, bool);
         void postOpenFile(std::string const&, bool);
         
         void preCloseFile(std::string const& lfn, bool primary);
         void postCloseFile(std::string const&, bool);

         void preModuleBeginStream(StreamContext const&, ModuleDescription const&);
         void postModuleBeginStream(StreamContext const&, ModuleDescription const&);

         void preModuleEndStream(StreamContext const&, ModuleDescription const&);
         void postModuleEndStream(StreamContext const&, ModuleDescription const&);

         void preGlobalBeginRun(GlobalContext const&);
         void postGlobalBeginRun(GlobalContext const&);

         void preGlobalEndRun(GlobalContext const&);
         void postGlobalEndRun(GlobalContext const&);

         void preStreamBeginRun(StreamContext const&);
         void postStreamBeginRun(StreamContext const&);

         void preStreamEndRun(StreamContext const&);
         void postStreamEndRun(StreamContext const&);

         void preGlobalBeginLumi(GlobalContext const&);
         void postGlobalBeginLumi(GlobalContext const&);

         void preGlobalEndLumi(GlobalContext const&);
         void postGlobalEndLumi(GlobalContext const&);

         void preStreamBeginLumi(StreamContext const&);
         void postStreamBeginLumi(StreamContext const&);

         void preStreamEndLumi(StreamContext const&);
         void postStreamEndLumi(StreamContext const&);

         void preEvent(StreamContext const&);
         void postEvent(StreamContext const&);

         void prePathEvent(StreamContext const&, PathContext const&);
         void postPathEvent(StreamContext const&, PathContext const&, HLTPathStatus const&);

         void preModuleConstruction(ModuleDescription const& md);
         void postModuleConstruction(ModuleDescription const& md);

         void preModuleBeginJob(ModuleDescription const& md);
         void postModuleBeginJob(ModuleDescription const& md);

         void preModuleEndJob(ModuleDescription const& md);
         void postModuleEndJob(ModuleDescription const& md);

         void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
         void postModuleEvent(StreamContext const&, ModuleCallingContext const&);
         
         void preModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
         void postModuleStreamBeginRun(StreamContext const&, ModuleCallingContext const&);
         void preModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);
         void postModuleStreamEndRun(StreamContext const&, ModuleCallingContext const&);

         void preModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
         void postModuleStreamBeginLumi(StreamContext const&, ModuleCallingContext const&);
         void preModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);
         void postModuleStreamEndLumi(StreamContext const&, ModuleCallingContext const&);

         void preModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
         void postModuleGlobalBeginRun(GlobalContext const&, ModuleCallingContext const&);
         void preModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);
         void postModuleGlobalEndRun(GlobalContext const&, ModuleCallingContext const&);

         void preModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
         void postModuleGlobalBeginLumi(GlobalContext const&, ModuleCallingContext const&);
         void preModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);
         void postModuleGlobalEndLumi(GlobalContext const&, ModuleCallingContext const&);

         void preSourceConstruction(ModuleDescription const& md);
         void postSourceConstruction(ModuleDescription const& md);

private:
         std::string indention_;
         unsigned int depth_;
         std::string dumpContextForLabel_;
         bool dumpNonModuleContext_;
      };
   }
}
#endif
