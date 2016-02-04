#ifndef DataFormats_Provenance_ProcessHistoryRegistry_h
#define DataFormats_Provenance_ProcessHistoryRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::ProcessHistoryID,edm::ProcessHistory> ProcessHistoryRegistry;
  typedef ProcessHistoryRegistry::collection_type ProcessHistoryMap;
  typedef ProcessHistoryRegistry::vector_type ProcessHistoryVector;
}

#endif
