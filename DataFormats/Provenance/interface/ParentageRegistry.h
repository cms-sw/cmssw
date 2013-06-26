#ifndef DataFormats_Provenance_ParentageRegistry_h
#define DataFormats_Provenance_ParentageRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageID.h"


// Note that this registry is *not* directly persistable. The contents
// are persisted, but not the container.
namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::ParentageID, edm::Parentage> ParentageRegistry;
  typedef ParentageRegistry::collection_type ParentageMap;
}

#endif
