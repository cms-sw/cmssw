#ifndef DataFormats_Provenance_EntryDescriptionRegistry_h
#define DataFormats_Provenance_EntryDescriptionRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"


// Note that this registry is *not* directly persistable. The contents
// are persisted, but not the container.
namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::EntryDescriptionID, edm::EntryDescription> EntryDescriptionRegistry;
  typedef EntryDescriptionRegistry::collection_type EntryDescriptionMap;
}

#endif
