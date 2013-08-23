#ifndef DataFormats_Provenance_ProcessHistoryRegistry_h
#define DataFormats_Provenance_ProcessHistoryRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/FullHistoryToReducedHistoryMap.h"

namespace edm {
  typedef edm::detail::ThreadSafeRegistry<edm::ProcessHistoryID,edm::ProcessHistory,edm::FullHistoryToReducedHistoryMap> ProcessHistoryRegistry;
  typedef ProcessHistoryRegistry::collection_type ProcessHistoryMap;
  typedef ProcessHistoryRegistry::vector_type ProcessHistoryVector;

  bool registerProcessHistory(ProcessHistory const& processHistory);
}

#endif
