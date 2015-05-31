// -*- C++ -*-
//
// Package:     LibraryLoader
// Class  :     FWLiteEnabler
//
// Implementation:
//     <Notes on implementation>
//
//

// system include files
#include <iostream>
#include "TROOT.h"
#include "TInterpreter.h"
#include "TApplication.h"

// user include files
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "FWCore/FWLite/src/BareRootProductGetter.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "FWCore/FWLite/interface/setRefStreamer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

  bool FWLiteEnabler::enabled_(false);


//
// constructors and destructor
//
// Note: this ctor will never be invoked.
// All data and functions are static
  FWLiteEnabler::FWLiteEnabler() {
  }


//
// member functions
//

  void
  FWLiteEnabler::enable() {
   if (enabled_) { return; }
   enabled_ = true;

   edmplugin::PluginManager::configure(edmplugin::standard::config());
   static BareRootProductGetter s_getter;
   //this function must be called
   // so that the TClass we need will be available
   fwlite::setRefStreamer(&s_getter);

   //Make it easy to load our headers
   TInterpreter* intrp= gROOT->GetInterpreter();
   char const* env = getenv("CMSSW_FWLITE_INCLUDE_PATH");
   if(0 != env) {
     //this is a comma separated list
     char const* start = env;
     char const* end;
     do {
       //find end
       for(end=start; *end!=0 and *end != ':';++end);
       std::string dir(start, end);
       intrp->AddIncludePath(dir.c_str());
       start = end+1;
     } while(*end != 0);
   }

   bool foundCMSIncludes = false;
   env = getenv("CMSSW_BASE");
   if(0 != env) {
     foundCMSIncludes = true;
     std::string dir(env);
     dir += "/src";
     intrp->AddIncludePath(dir.c_str());
   }

   env = getenv("CMSSW_RELEASE_BASE");
   if(0 != env) {
     foundCMSIncludes = true;
     std::string dir(env);
     dir += "/src";
     intrp->AddIncludePath(dir.c_str());
   }
   if(not foundCMSIncludes) {
     std::cerr <<"Could not find the environment variables \n"
     <<"  CMSSW_BASE or\n"
     <<"  CMSSW_RELEASE_BASE\n"
     <<" therefore attempting to '#include' any CMS headers will not work"<<std::endl;
   }
   if (0 != gApplication) {
     gApplication->InitializeGraphics();
   }
  }

