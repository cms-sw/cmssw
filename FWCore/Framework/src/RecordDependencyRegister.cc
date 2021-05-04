// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     RecordDependencyRegister
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  root
//         Created:  Thu, 26 Apr 2018 16:03:47 GMT
//

// system include files
#include "tbb/concurrent_unordered_map.h"

// user include files
#include "FWCore/Framework/interface/RecordDependencyRegister.h"

namespace edm {
  namespace eventsetup {
    namespace {
      struct KeyHash {
        std::size_t operator()(EventSetupRecordKey const& iKey) const { return iKey.type().value().hash_code(); }
      };

      tbb::concurrent_unordered_map<EventSetupRecordKey, DepFunction, KeyHash>& getMap() {
        static tbb::concurrent_unordered_map<EventSetupRecordKey, DepFunction, KeyHash> s_map;
        return s_map;
      }

      tbb::concurrent_unordered_map<EventSetupRecordKey, bool, KeyHash>& getAllowMap() {
        static tbb::concurrent_unordered_map<EventSetupRecordKey, bool, KeyHash> s_allow_map;
        return s_allow_map;
      }
    }  // namespace

    std::set<EventSetupRecordKey> dependencies(EventSetupRecordKey const& iKey) {
      auto& map = getMap();
      auto itFind = map.find(iKey);
      if (itFind != map.end()) {
        return itFind->second();
      }
      return std::set<EventSetupRecordKey>();
    }

    bool allowConcurrentIOVs(EventSetupRecordKey const& iKey) {
      auto& map = getAllowMap();
      auto itFind = map.find(iKey);
      if (itFind != map.end()) {
        return itFind->second;
      }
      return false;
    }

    void addDependencyFunction(EventSetupRecordKey iKey, DepFunction iFunction, bool allowConcurrentIOVs) {
      getMap().emplace(iKey, iFunction);
      getAllowMap().emplace(iKey, allowConcurrentIOVs);
    }

  }  // namespace eventsetup
}  // namespace edm
