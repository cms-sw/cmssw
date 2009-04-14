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
//
#include "FWCore/PluginManager/interface/PluginFactory.h"


namespace cond {
  class DataProxyWrapperBase;
  class Connection;

  typedef edmplugin::PluginFactory< cond::DataProxyWrapperBase* ( cond::Connection&, std::string const&, std::string const& ) > ProxyFactory;

   const char* pluginCategory();
}

// compatibility mode
namespace oldcond {
  class Connection;

typedef edmplugin::PluginFactory< edm::eventsetup::DataProxy* ( cond::Connection*, std::map<std::string,std::string>::iterator& ) > ProxyFactory;

  // const char* pluginCategory();
}

#endif /* CONDCORE_PLUGINSYSTEM_PROXYFACTORY_H */
