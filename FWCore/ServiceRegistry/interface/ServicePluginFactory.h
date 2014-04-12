#ifndef ServiceRegistry_ServicePluginFactory_h
#define ServiceRegistry_ServicePluginFactory_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServicePluginFactory
// 
/**\class ServicePluginFactory ServicePluginFactory.h FWCore/ServiceRegistry/interface/ServicePluginFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:00 EDT 2005
//

// system include files
#include "FWCore/PluginManager/interface/PluginFactory.h"

// user include files

// forward declarations
namespace edm {
   namespace serviceregistry {
      class ServiceMakerBase;
      
     typedef edmplugin::PluginFactory< ServiceMakerBase* ()> ServicePluginFactory;
   }
}

#endif
