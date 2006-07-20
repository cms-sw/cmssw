#ifndef FWCoreFrameworkModuleDescriptionRegistry_h
#define FWCoreFrameworkModuleDescriptionRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/ModuleDescriptionID.h"

namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::ModuleDescriptionID, edm::ModuleDescription> ModuleDescriptionRegistry;
  typedef ModuleDescriptionRegistry::collection_type ModuleDescriptionMap;
}

#endif
