#ifndef DataFormats_Provenance_EntryDescriptionRegistry_h
#define DataFormats_Provenance_EntryDescriptionRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"


// Note that this registry is *not* directly persistable. The contents
// are persisted, but not the container.
namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::EntryDescriptionID, edm::EventEntryDescription> EntryDescriptionRegistry;
  typedef EntryDescriptionRegistry::collection_type EntryDescriptionMap;
}

#endif
