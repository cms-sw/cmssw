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
// $Id: AutoLibraryLoader.cc,v 1.2 2006/05/29 13:03:24 chrjones Exp $
//

// system include files
#include <iostream>
#include "TROOT.h"

// user include files
#include "FWCore/FWLite/src/AutoLibraryLoader.h"
#include "FWCore/FWLite/src/stdNamespaceAdder.h"
#include "FWCore/FWLite/src/BareRootProductGetter.h"

#include "PluginManager/PluginManager.h"
#include "PluginManager/ModuleCache.h"
#include "PluginManager/Module.h"
#include "PluginManager/PluginCapabilities.h"

#include "Reflex/Type.h"
#include "Cintex/Cintex.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AutoLibraryLoader::AutoLibraryLoader()
{
   seal::PluginManager::get()->initialise();
   gROOT->AddClassGenerator(this);
   ROOT::Cintex::Cintex::Enable();
}


//
// member functions
//

TClass *
AutoLibraryLoader::GetClass(const char* classname, Bool_t load)
{
   TClass* returnValue = 0;
//   std::cout <<"looking for "<<classname <<" load "<<(load? "T":"F")<< std::endl;
   if(load) {
      static const std::string cPrefix("LCGReflex/");
      //std::cout <<"asking to find "<<cPrefix+classname<<std::endl;
      seal::PluginCapabilities::get()->load(cPrefix+classname);

      ROOT::Reflex::Type t = ROOT::Reflex::Type::ByName(classname);
      if(ROOT::Reflex::Type() != t ) {
	 //std::cout <<"loaded "<<classname<<std::endl;
	 return gROOT->GetClass(classname,kFALSE);
      } else {
	 //see if adding a std namespace helps
	 std::string name = fwlite::stdNamespaceAdder(classname);

	 seal::PluginCapabilities::get()->load(cPrefix+name);

	 ROOT::Reflex::Type t = ROOT::Reflex::Type::ByName(classname);
	 if(ROOT::Reflex::Type() != t ) {
	    //std::cout <<"loaded "<<classname<<std::endl;
	    return gROOT->GetClass(classname,kFALSE);
	 }
      }
   }
   return returnValue;
}


TClass *
AutoLibraryLoader::GetClass(const type_info& typeinfo, Bool_t load)
{
   //std::cout <<"looking for type "<<typeinfo.name()<<std::endl;
   TClass* returnValue = 0;
   if(load){
      return GetClass(typeinfo.name(), load);
   }
   return returnValue;
}

void
AutoLibraryLoader::enable()
{
   static BareRootProductGetter s_getter;
   static edm::EDProductGetter::Operate s_op(&s_getter);
   static AutoLibraryLoader s_loader;
}




void
AutoLibraryLoader::loadAll()
{
  // std::cout <<"LoadAllDictionaries"<<std::endl;
  enable();
  
  seal::PluginManager                       *db =  seal::PluginManager::get();
  seal::PluginManager::DirectoryIterator    dir;
  seal::ModuleCache::Iterator               plugin;
  seal::ModuleDescriptor                    *cache;
  unsigned                            i;
  
  const std::string mycat("Capability");
  
  for (dir = db->beginDirectories(); dir != db->endDirectories(); ++dir) {
    for (plugin = (*dir)->begin(); plugin != (*dir)->end(); ++plugin) {
      for (cache=(*plugin)->cacheRoot(), i=0; i < cache->children(); ++i) {
        //std::cout <<" "<<cache->child(i)->token(0)<<std::endl;
        if (cache->child(i)->token(0) == mycat) {
          const std::string cap = cache->child(i)->token(1);
          //std::cout <<"  "<<cap<<std::endl;
          // check that cap starts with either LCGDict or LCGReflex (not really required)
          static const std::string cPrefix("LCGReflex/");
          if(cPrefix == cap.substr(cPrefix.size())) {
            seal::PluginCapabilities::get()->load(cap);
          }
          break;
        }
      }
    }
  }
}


ClassImp(AutoLibraryLoader);
