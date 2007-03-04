#ifndef DataFormats_Provenance_ModuleDescriptionRegistry_h
#define DataFormats_Provenance_ModuleDescriptionRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"

namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::ModuleDescriptionID, edm::ModuleDescription> ModuleDescriptionRegistry;
  typedef ModuleDescriptionRegistry::collection_type ModuleDescriptionMap;
}

#endif
