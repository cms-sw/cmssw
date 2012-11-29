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
// $Id: Tracer.h,v 1.12 2010/01/19 22:37:06 wdd Exp $
//

// system include files

// user include files

// forward declarations

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"


namespace edm {
   class ConfigurationDescriptions;

   namespace service {
      class Tracer {
public:
         Tracer(const ParameterSet&,ActivityRegistry&);
         
         static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

         void postBeginJob();
         void postEndJob();
         
         void preBeginRun(RunID const& id, Timestamp const& ts);
         void postBeginRun(Run const& run, EventSetup const& es);

         void preBeginLumi(LuminosityBlockID const& id, Timestamp const& ts);
         void postBeginLumi(LuminosityBlock const& run, EventSetup const& es);

         void preEvent(EventID const& id, Timestamp const& ts);
         void postEvent(Event const& ev, EventSetup const& es);
         
         void preEndLumi(LuminosityBlockID const& id, Timestamp const& ts);
         void postEndLumi(LuminosityBlock const& run, EventSetup const& es);

         void preEndRun(RunID const& id, Timestamp const& ts);
         void postEndRun(Run const& run, EventSetup const& es);

         void preSourceConstruction(ModuleDescription const& md);
         void postSourceConstruction(ModuleDescription const& md);

         void preModuleConstruction(ModuleDescription const& md);
         void postModuleConstruction(ModuleDescription const& md);

         void preModuleBeginJob(ModuleDescription const& md);
         void postModuleBeginJob(ModuleDescription const& md);

         void preModuleBeginRun(ModuleDescription const& md);
         void postModuleBeginRun(ModuleDescription const& md);

         void preModuleBeginLumi(ModuleDescription const& md);
         void postModuleBeginLumi(ModuleDescription const& md);

         void preModuleEvent(ModuleDescription const& md);
         void postModuleEvent(ModuleDescription const& md);
         
         void preModuleEndLumi(ModuleDescription const& md);
         void postModuleEndLumi(ModuleDescription const& md);

         void preModuleEndRun(ModuleDescription const& md);
         void postModuleEndRun(ModuleDescription const& md);

         void preModuleEndJob(ModuleDescription const& md);
         void postModuleEndJob(ModuleDescription const& md);

         void preSourceEvent();
         void postSourceEvent();

         void preSourceLumi();
         void postSourceLumi();

         void preSourceRun();
         void postSourceRun();

         void preOpenFile();
         void postOpenFile();
         
         void preCloseFile(std::string const& lfn, bool primary);
         void postCloseFile();
         
         void prePathBeginRun(std::string const& s);
         void postPathBeginRun(std::string const& s, HLTPathStatus const& hlt);

         void prePathBeginLumi(std::string const& s);
         void postPathBeginLumi(std::string const& s, HLTPathStatus const& hlt);

         void prePathEvent(std::string const& s);
         void postPathEvent(std::string const& s, HLTPathStatus const& hlt);

         void prePathEndLumi(std::string const& s);
         void postPathEndLumi(std::string const& s, HLTPathStatus const& hlt);

         void prePathEndRun(std::string const& s);
         void postPathEndRun(std::string const& s, HLTPathStatus const& hlt);

private:
         std::string indention_;
         unsigned int depth_;
         
      };
   }
}
   
#endif
