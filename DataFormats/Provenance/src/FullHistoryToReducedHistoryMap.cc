#include "DataFormats/Provenance/interface/FullHistoryToReducedHistoryMap.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  FullHistoryToReducedHistoryMap::FullHistoryToReducedHistoryMap() {
    // Put in the conversion for the ID of an empty process history.
    // It always maps onto itself, often is needed, and gives
    // us something valid to initialize the iterator with. That removes
    // the need to check for its validity and also for the validity
    // of the fullID argument to the reduce function, because the invalid
    // ProcessHistoryID is strangely also defined as the ID of the empty
    // Process History (maybe that should be fixed ...).
    ProcessHistory ph;
    ph.setProcessHistoryID();
    std::pair<ProcessHistoryID, ProcessHistoryID> newEntry(ph.id(), ph.id());
    std::pair<Map::iterator, bool> result = cache_.insert(newEntry);
    previous_ = result.first;
  }

  ProcessHistoryID const&
  FullHistoryToReducedHistoryMap::reduceProcessHistoryID(ProcessHistoryID const& fullID) {
    if (previous_->first == fullID) return previous_->second;
    Map::const_iterator iter = cache_.find(fullID);
    if (iter != cache_.end()) {
      previous_ = iter;
      return iter->second;
    }
    ProcessHistoryRegistry* registry = ProcessHistoryRegistry::instance();
    ProcessHistory ph;
    if (!registry->getMapped(fullID, ph)) {
      throw Exception(errors::LogicError)
        << "FullHistoryToReducedHistoryMap::reduceProcessHistoryID\n"
        << "ProcessHistory not found in registry\n"
        << "Contact a Framework developer\n";
    }
    ph.reduce();
    std::pair<ProcessHistoryID, ProcessHistoryID> newEntry(fullID, ph.setProcessHistoryID());
    std::pair<Map::iterator, bool> result = cache_.insert(newEntry);
    previous_ = result.first;
    return result.first->second;
  }
}
