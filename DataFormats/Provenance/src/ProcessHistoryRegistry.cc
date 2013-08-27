#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/FullHistoryToReducedHistoryMap.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.icc"

namespace edm {
  bool
  registerProcessHistory(ProcessHistory const& processHistory) {
    ProcessHistoryRegistry& registry = *ProcessHistoryRegistry::instance();
    bool newlyAdded = registry.insertMapped(processHistory);
    if(newlyAdded) {
      registry.extraForUpdate().insertMapping(processHistory); 
    }
    return newlyAdded;
  }
}

DEFINE_THREAD_SAFE_REGISTRY_INSTANCE(ProcessHistoryRegistry)
