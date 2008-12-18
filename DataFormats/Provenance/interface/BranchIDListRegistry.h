#ifndef FWCore_Framework_BranchIDListRegistry_h
#define FWCore_Framework_BranchIDListRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeIndexedRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"

namespace edm {
  typedef edm::detail::ThreadSafeIndexedRegistry<BranchIDList, BranchIDListHelper> BranchIDListRegistry;
  typedef BranchIDListRegistry::collection_type BranchIDLists;
}

#endif
