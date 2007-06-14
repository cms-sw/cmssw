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
// $Id: Tracer.h,v 1.8 2007/01/09 17:33:06 chrjones Exp $
//

// system include files

// user include files

// forward declarations

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"


namespace edm {
   namespace service {
      class Tracer {
public:
         Tracer(const ParameterSet&,ActivityRegistry&);
         
         void postBeginJob();
         void postEndJob();
         
         void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
         void postEventProcessing(const Event&, const EventSetup&);
         
         void preModuleConstruction(const ModuleDescription&);
         void postModuleConstruction(const ModuleDescription&);

         void preModuleBeginJob(const ModuleDescription&);
         void postModuleBeginJob(const ModuleDescription&);

         void preModule(const ModuleDescription&);
         void postModule(const ModuleDescription&);
         
         void preModuleEndJob(const ModuleDescription&);
         void postModuleEndJob(const ModuleDescription&);

         void preSource();
         void postSource();
         
         void prePath(const std::string&);
         void postPath(const std::string&, const edm::HLTPathStatus&);
private:
         std::string indention_;
         unsigned int depth_;
         
      };
   }
}



#endif
