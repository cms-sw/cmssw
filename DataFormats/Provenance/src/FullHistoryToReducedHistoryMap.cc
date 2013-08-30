#include "DataFormats/Provenance/interface/FullHistoryToReducedHistoryMap.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  FullHistoryToReducedHistoryMap::FullHistoryToReducedHistoryMap() {
    // We need to map the empty process history ID.
    insertMapping(ProcessHistory());
  }

  ProcessHistoryID const&
  FullHistoryToReducedHistoryMap::reducedProcessHistoryID(ProcessHistoryID const& fullID) const {
    Map::const_iterator iter = fullHistoryToReducedHistoryMap_.find(fullID);
    assert(iter != fullHistoryToReducedHistoryMap_.end());
    return iter->second;
  }

  void
  FullHistoryToReducedHistoryMap::insertMapping(ProcessHistory const& processHistory) {
    ProcessHistory ph(processHistory);
    ph.reduce();
    std::pair<ProcessHistoryID, ProcessHistoryID> newEntry(processHistory.id(), ph.setProcessHistoryID());
    fullHistoryToReducedHistoryMap_.insert(newEntry);
  }
}
