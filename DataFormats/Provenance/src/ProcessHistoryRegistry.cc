#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/processingOrderMerge.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>
namespace edm {
  ProcessHistoryRegistry::ProcessHistoryRegistry() : data_(), extra_() {
    // insert the mapping for an empty process history
    extra_.insert(std::pair<ProcessHistoryID, ProcessHistoryID>(ProcessHistory().id(), ProcessHistory().reduce().id()));
  }

  bool ProcessHistoryRegistry::registerProcessHistory(ProcessHistory const& processHistory) {
    //make sure the process history ID is cached
    auto tmp = processHistory;
    ProcessHistoryID id = tmp.setProcessHistoryID();
    bool newlyAdded = (data_.find(id) == data_.end());
    if (newlyAdded) {
      data_.emplace(id, tmp);
      extra_.emplace(std::move(id), tmp.reduce().id());
    }
    return newlyAdded;
  }

  ProcessHistoryID const& ProcessHistoryRegistry::reducedProcessHistoryID(ProcessHistoryID const& fullID) const {
    auto const& iter = extra_.find(fullID);
    assert(iter != extra_.end());
    return iter->second;
  }

  bool ProcessHistoryRegistry::getMapped(ProcessHistoryID const& key, ProcessHistory& value) const {
    auto const& iter = data_.find(key);
    bool found = (iter != data_.end());
    if (found) {
      value = iter->second;
    }
    return found;
  }

  ProcessHistory const* ProcessHistoryRegistry::getMapped(ProcessHistoryID const& key) const {
    auto const& iter = data_.find(key);
    if (iter == data_.end()) {
      return nullptr;
    }
    return &iter->second;
  }

  void processingOrderMerge(ProcessHistoryRegistry const& registry, std::vector<std::string>& processNames) {
    /*handle a case for reading an old file merge in a possibly incompatible way*/
    if (registry.begin() == registry.end()) {
      return;
    }
    auto itLongest = registry.begin();
    for (auto it = registry.begin(), itEnd = registry.end(); it != itEnd; ++it) {
      if (it->second.size() > itLongest->second.size()) {
        itLongest = it;
      }
    }
    processingOrderMerge(itLongest->second, processNames);
    for (auto it = registry.begin(), itEnd = registry.end(); it != itEnd; ++it) {
      if (it == itLongest) {
        continue;
      }
      processingOrderMerge(it->second, processNames);
    }
  }

}  // namespace edm
