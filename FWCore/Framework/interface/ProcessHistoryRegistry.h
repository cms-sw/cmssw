#ifndef FWCoreFrameworkProcessHistoryRegistry_h
#define FWCoreFrameworkProcessHistoryRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Common/interface/ProcessHistory.h"
#include "DataFormats/Common/interface/ProcessHistoryID.h"

namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::ProcessHistoryID,edm::ProcessHistory> ProcessHistoryRegistry;
  typedef ProcessHistoryRegistry::collection_type ProcessHistoryMap;
}

#endif
