#ifndef CONDCORE_PLUGINSYSTEM_PROXYFACTORY_H
#define CONDCORE_PLUGINSYSTEM_PROXYFACTORY_H
// -*- C++ -*-
//
// Package:     PluginSystem
// Class  :     ProxyFactory
// 
/**\class ProxyFactory ProxyFactory.h CondCore/PluginSystem/interface/ProxyFactory.h
   
Description: <one line class summary>

Usage:
<usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 19:14:06 EDT 2005
// $Id: ProxyFactory.h,v 1.7 2007/04/10 23:02:52 wmtan Exp $
//

// system include files

// user include files
//#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
// forward declarations
/*namespace pool{
  class IDataSvc;
}
*/
namespace edm {
  namespace eventsetup {
    class DataProxy;
  }
}

namespace cond {
  class PoolStorageManager;

typedef edmplugin::PluginFactory< 
   edm::eventsetup::DataProxy* ( cond::PoolStorageManager*, std::map<std::string,std::string>::iterator& ) > 
        ProxyFactory;

   const char* pluginCategory();
}

#endif /* CONDCORE_PLUGINSYSTEM_PROXYFACTORY_H */
