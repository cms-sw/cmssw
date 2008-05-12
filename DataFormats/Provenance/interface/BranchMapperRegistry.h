#ifndef DataFormats_Provenance_BranchMapperRegistry_h
#define DataFormats_Provenance_BranchMapperRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/BranchMapperID.h"


// Note that this registry is *not* directly persistable. The contents
// are persisted, but not the container.
namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::BranchMapperID, edm::BranchMapper> BranchMapperRegistry;
  typedef BranchMapperRegistry::collection_type BranchMapperMap;
}

#endif
