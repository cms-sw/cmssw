#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>
namespace edm {
  ProcessHistoryRegistry::ProcessHistoryRegistry() : data_(), extra_() {
    // insert the mapping for an empty process history
    extra_.insert(std::pair<ProcessHistoryID, ProcessHistoryID>(ProcessHistory().id(), ProcessHistory().reduce().id()));
  }

  bool
  ProcessHistoryRegistry::registerProcessHistory(ProcessHistory const& processHistory) {
    ProcessHistoryID id = processHistory.id();
    bool newlyAdded = (data_.find(id) == data_.end());
    if(newlyAdded) {
      data_.insert(std::pair<ProcessHistoryID, ProcessHistory>(id, processHistory));
      ProcessHistory toBeReduced(processHistory);
      extra_.insert(std::pair<ProcessHistoryID, ProcessHistoryID>(id, toBeReduced.reduce().id()));
    }
    return newlyAdded;
  }

  ProcessHistoryID const&
  ProcessHistoryRegistry::reducedProcessHistoryID(ProcessHistoryID const& fullID) const {
    auto const& iter = extra_.find(fullID);
    assert(iter != extra_.end());
    return iter->second;
  }

  bool
  ProcessHistoryRegistry::getMapped(ProcessHistoryID const& key, ProcessHistory& value) const {
    auto const& iter = data_.find(key);
    bool found = (iter != data_.end());
    if(found) {
      value = iter->second;
    }
    return found;
  }

  ProcessHistory const*
  ProcessHistoryRegistry::getMapped(ProcessHistoryID const& key) const {
    auto const& iter = data_.find(key);
    if(iter == data_.end()) {
      return nullptr; 
    }
    return &iter->second;
  }
}
