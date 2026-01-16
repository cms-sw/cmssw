#ifndef STORAGE_FACTORY_STORAGE_MAKER_FACTORY_H
#define STORAGE_FACTORY_STORAGE_MAKER_FACTORY_H

#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm::storage {
  class StorageMaker;
  typedef edmplugin::PluginFactory<StorageMaker *()> StorageMakerFactory;
}  // namespace edm::storage
#endif  // STORAGE_FACTORY_STORAGE_MAKER_FACTORY_H
