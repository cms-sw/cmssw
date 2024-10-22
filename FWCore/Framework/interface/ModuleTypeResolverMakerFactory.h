#ifndef FWCore_Framework_itnerface_ModuleTypeResolverFactory_h
#define FWCore_Framework_itnerface_ModuleTypeResolverFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include <string>
#include <vector>

namespace edm {
  class ModuleTypeResolverMaker;

  using ModuleTypeResolverMakerFactory = edmplugin::PluginFactory<ModuleTypeResolverMaker const*()>;
}  // namespace edm

#endif
