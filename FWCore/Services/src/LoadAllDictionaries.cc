// -*- C++ -*-
//
// Package:     Services
// Class  :     LoadAllDictionaries
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Sep 15 09:47:48 EDT 2005
// $Id: LoadAllDictionaries.cc,v 1.4 2006/07/07 16:04:37 wmtan Exp $
//

// system include files
#include "Cintex/Cintex.h"

// user include files
#include "FWCore/Services/src/LoadAllDictionaries.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PluginManager/PluginManager.h"
#include "PluginManager/ModuleCache.h"
#include "PluginManager/Module.h"
#include "PluginManager/PluginCapabilities.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
edm::service::LoadAllDictionaries::LoadAllDictionaries(const edm::ParameterSet& iConfig)
{
   bool doLoad(iConfig.getUntrackedParameter("doLoad",true));
   if(doLoad) {

    ROOT::Cintex::Cintex::Enable();

      seal::PluginManager                       *db =  seal::PluginManager::get();
      seal::PluginManager::DirectoryIterator    dir;
      seal::ModuleCache::Iterator               plugin, pluginEnd;
      seal::ModuleDescriptor                    *cache;
      unsigned                            i;
      
      
      // std::cout <<"LoadAllDictionaries"<<std::endl;
      
      const std::string mycat("Capability");
      const std::string mystring("edm::Wrapper");
      
      for (dir = db->beginDirectories(); dir != db->endDirectories(); ++dir) {
         for (plugin = (*dir)->begin(), pluginEnd = (*dir)->end(); plugin != pluginEnd; ++plugin) {
            for (cache=(*plugin)->cacheRoot(), i=0; i < cache->children(); ++i) {
               //std::cout <<" "<<cache->child(i)->token(0)<<std::endl;
               if (cache->child(i)->token(0) == mycat) {
                  const std::string cap = cache->child(i)->token(1);
                  //std::cout <<"  "<<cap<<std::endl;
                  // check that cap starts with either LCGDict or LCGReflex (not really required)
		  if (cap.find(mystring) != std::string::npos) { 
                    seal::PluginCapabilities::get()->load(cap);
                  }
               }
            }
         }
      }
   }
}

