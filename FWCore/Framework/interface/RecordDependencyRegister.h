#ifndef FWCore_Framework_RecordDependencyRegister_h
#define FWCore_Framework_RecordDependencyRegister_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     RecordDependencyRegister
//
/**\class RecordDependencyRegister RecordDependencyRegister.h "RecordDependencyRegister.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Thu, 26 Apr 2018 15:46:36 GMT
//

// system include files
#include <set>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/findDependentRecordsFor.h"

// forward declarations

namespace edm {
  namespace eventsetup {
    using DepFunction = std::set<EventSetupRecordKey> (*)();

    std::set<EventSetupRecordKey> dependencies(EventSetupRecordKey const&);
    bool allowConcurrentIOVs(EventSetupRecordKey const&);

    void addDependencyFunction(EventSetupRecordKey iKey, DepFunction iFunction, bool allowConcurrentIOVs);

    template <typename T>
    struct RecordDependencyRegister {
      RecordDependencyRegister() {
        addDependencyFunction(EventSetupRecordKey::makeKey<T>(), &findDependentRecordsFor<T>, T::allowConcurrentIOVs_);
      }
    };
  }  // namespace eventsetup
}  // namespace edm

#endif
