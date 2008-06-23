// -*- C++ -*-
//
// Package:     LibraryLoader
// Class  :     AutoLibraryLoader
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed Nov 30 14:55:01 EST 2005
// $Id: AutoLibraryLoader.cc,v 1.21 2008/06/12 22:17:22 dsr Exp $
//

// system include files
#include <iostream>
#include "TROOT.h"
#include "TInterpreter.h"
#include "TApplication.h"

// user include files
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
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

bool AutoLibraryLoader::enabled_(false);

//
// constructors and destructor
//
AutoLibraryLoader::AutoLibraryLoader()
{
}


//
// member functions
//

void
AutoLibraryLoader::enable()
{
   if (enabled_) { return; }
   enabled_ = true;

   edmplugin::PluginManager::configure(edmplugin::standard::config());
   static BareRootProductGetter s_getter;
   static edm::EDProductGetter::Operate s_op(&s_getter);
   edm::RootAutoLibraryLoader::enable();
   //this function must be called after enabling the autoloader
   // so that the Reflex dictionaries will be converted to ROOT 
   // dictionaries and the TClass we need will be available
   fwlite::setRefStreamer(&s_getter);
   
   //Make it easy to load our headers
   TInterpreter* intrp= gROOT->GetInterpreter();
   const char* env = getenv("CMSSW_FWLITE_INCLUDE_PATH");
   if( 0 != env) {
     //this is a comma separated list
     const char* start = env;
     const char* end = env;
     do{
       //find end
       for(end=start; *end!=0 and *end != ':';++end);
       std::string dir(start, end);
       intrp->AddIncludePath(dir.c_str());
       start = end+1;
     }while(*end != 0);
   }
   
   bool foundCMSIncludes = false;
   env = getenv("CMSSW_BASE");
   if( 0 != env) {
     foundCMSIncludes = true;
     std::string dir(env);
     dir += "/src";
     intrp->AddIncludePath(dir.c_str());
   }

   env = getenv("CMSSW_RELEASE_BASE");
   if( 0 != env) {
     foundCMSIncludes = true;
     std::string dir(env);
     dir += "/src";
     intrp->AddIncludePath(dir.c_str());
   }
   if( not foundCMSIncludes) {
     std::cerr <<"Could not find the environment variables \n"
     <<"  CMSSW_BASE or\n"
     <<"  CMSSW_RELEASE_BASE\n"
     <<" therefore attempting to '#include' any CMS headers will not work"<<std::endl;
   }
   if (0 != gApplication) {
     gApplication->InitializeGraphics();
   }
}

void
AutoLibraryLoader::loadAll()
{
  // std::cout <<"LoadAllDictionaries"<<std::endl;
  enable();
  edm::RootAutoLibraryLoader::loadAll();
}


ClassImp(AutoLibraryLoader)
