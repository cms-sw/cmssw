#ifndef CONDCORE_PLUGINSYSTEM_PRODUCTRESOLVERFACTORY_H
#define CONDCORE_PLUGINSYSTEM_PRODUCTRESOLVERFACTORY_H
// -*- C++ -*-
//
// Package:     ESSources
// Class  :     ProductResolverFactory
//
/**\class ProductResolverFactory ProductResolverFactory.h CondCore/ESSources/interface/ProductResolverFactory.h
   
Description: <one line class summary>

Usage:
<usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 19:14:06 EDT 2005
//
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include <string>

namespace cond {
  class ProductResolverWrapperBase;
  namespace persistency {
    class Session;
  }

  typedef edmplugin::PluginFactory<cond::ProductResolverWrapperBase*()> ProductResolverFactory;

  const char* pluginCategory();
}  // namespace cond

#endif /* CONDCORE_PLUGINSYSTEM_PRODUCTRESOLVERFACTORY_H */
