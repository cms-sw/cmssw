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
// $Id$
//

// system include files
#include <iostream>

// user include files
#include "FWCore/Services/src/LoadAllDictionaries.h"
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
edm::service::LoadAllDictionaries::LoadAllDictionaries()
{
   seal::PluginManager                       *db =  seal::PluginManager::get ();
   seal::PluginManager::DirectoryIterator    dir;
   seal::ModuleCache::Iterator               plugin;
   seal::ModuleDescriptor                    *cache;
   unsigned                            i;
   
   
   //std::cout <<"LoadAllDictionaries"<<std::endl;
   
   const std::string mycat("Capability");
   
   for (dir = db->beginDirectories (); dir != db->endDirectories (); ++dir) {
      for (plugin = (*dir)->begin (); plugin != (*dir)->end (); ++plugin) {
         for (cache=(*plugin)->cacheRoot(), i=0; i < cache->children(); ++i) {
            //std::cout <<" "<<cache->child(i)->token(0)<<std::endl;
            if (cache->child (i)->token (0)==mycat) {
               const std::string cap = cache->child (i)->token (1);
               //std::cout <<"  "<<cap<<std::endl;
               // check that cap starts with either LCGDict or LCGReflex (not really required)
               seal::PluginCapabilities::get()->load(cap);
            }
         }
      }
   }
}

