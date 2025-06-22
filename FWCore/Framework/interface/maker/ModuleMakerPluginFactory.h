#ifndef FWCore_Framework_ModuleMakerPluginFactory_h
#define FWCore_Framework_ModuleMakerPluginFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  class ModuleMakerBase;

  using ModuleMakerPluginFactory = edmplugin::PluginFactory<ModuleMakerBase*()>;
}  // namespace edm

#endif
