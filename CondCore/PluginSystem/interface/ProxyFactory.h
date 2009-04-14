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
// $Id: ProxyFactory.h,v 1.9 2007/08/23 11:33:07 xiezhen Exp $
//
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  namespace eventsetup {
    class DataProxy;
  }
}

namespace cond {
  class Connection;

typedef edmplugin::PluginFactory< edm::eventsetup::DataProxy* ( cond::Connection*, std::map<std::string,std::string>::iterator& ) > ProxyFactory;

   const char* pluginCategory();
}

// compatibility mode
namespace oldcond {
  class Connection;

typedef edmplugin::PluginFactory< edm::eventsetup::DataProxy* ( cond::Connection*, std::map<std::string,std::string>::iterator& ) > ProxyFactory;

   const char* pluginCategory();
}

#endif /* CONDCORE_PLUGINSYSTEM_PROXYFACTORY_H */
