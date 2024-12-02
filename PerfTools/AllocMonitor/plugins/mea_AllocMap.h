#ifndef PerfTools_AllocMonitor_mea_AllocMap_h
#define PerfTools_AllocMonitor_mea_AllocMap_h
// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     AllocMap
//
/**\class mea_AllocMap mea_AllocMap.h "PerfTools/AllocMonitor/interface/mea_AllocMap.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 11 Oct 2024 19:15:46 GMT
//

// system include files
#include <vector>
#include <cassert>

// user include files

// forward declarations

namespace edm::service::moduleEventAlloc {
  class AllocMap {
  public:
    AllocMap() = default;
    AllocMap(AllocMap const&) = default;
    AllocMap& operator=(AllocMap const&) = default;
    AllocMap(AllocMap&&) = default;
    AllocMap& operator=(AllocMap&&) = default;

    // ---------- const member functions ---------------------
    std::size_t size() const { return keys_.size(); }

    //returns size() if not here
    std::size_t findOffset(void const* iKey) const {
      auto bound = std::lower_bound(keys_.begin(), keys_.end(), iKey);
      if (bound == keys_.end() or *bound != iKey) {
        return size();
      }
      return bound - keys_.begin();
    }

    std::vector<std::size_t> const& allocationSizes() const { return values_; }
    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void insert(void const* iKey, std::size_t iValue) {
      auto offset = insertOffset(iKey);
      if (offset != size() and keys_[offset] == iKey) {
        values_[offset] = iValue;
        return;
      }
      keys_.insert(keys_.begin() + offset, iKey);
      values_.insert(values_.begin() + offset, iValue);
    }
    //returns 0 if not here else returns allocation size
    std::size_t erase(void const* iKey) {
      assert(keys_.size() == values_.size());
      auto offset = findOffset(iKey);
      if (offset == size()) {
        return 0;
      }
      auto v = values_[offset];
      values_.erase(values_.begin() + offset);
      keys_.erase(keys_.begin() + offset);

      return v;
    }
    void clearSizes() {
      values_.clear();
      values_.shrink_to_fit();
    }

    void clear() {
      clearSizes();
      keys_.clear();
      keys_.shrink_to_fit();
    }

  private:
    // ---------- member data --------------------------------
    std::size_t insertOffset(void const* key) const {
      auto bound = std::lower_bound(keys_.begin(), keys_.end(), key);
      return bound - keys_.begin();
    }

    std::vector<void const*> keys_;
    std::vector<std::size_t> values_;
  };
}  // namespace edm::service::moduleEventAlloc
#endif
