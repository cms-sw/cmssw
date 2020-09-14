#ifndef FWCore_Framework_findDependentRecordsFor_h
#define FWCore_Framework_findDependentRecordsFor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     findDependentRecordsFor
//
/**\function findDependentRecordsFor findDependentRecordsFor.h "findDependentRecordsFor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 26 Apr 2018 14:13:06 GMT
//

// system include files
#include <set>
#include <type_traits>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DependentRecordTag.h"
#include "FWCore/Utilities/interface/mplVector.h"

// forward declarations
namespace edm {
  namespace eventsetup {

    //Recursively desend container while adding first type's info to set
    template <typename TFirst, typename TRemaining>
    void addRecordToDependencies(const TFirst*, const TRemaining*, std::set<EventSetupRecordKey>& oSet) {
      oSet.insert(EventSetupRecordKey::makeKey<TFirst>());
      using Pop = edm::mpl::Pop<TRemaining>;
      if constexpr (not Pop::empty) {
        const typename Pop::Item* next(nullptr);
        const typename Pop::Remaining* remaining(nullptr);
        addRecordToDependencies(next, remaining, oSet);
      }
    }

    template <typename T>
    std::set<EventSetupRecordKey> findDependentRecordsFor() {
      std::set<EventSetupRecordKey> returnValue;
      if constexpr (std::is_base_of_v<edm::eventsetup::DependentRecordTag, T>) {
        using list_type = typename T::list_type;
        using Pop = edm::mpl::Pop<list_type>;

        const typename Pop::Item* begin(nullptr);
        const typename Pop::Remaining* remaining(nullptr);
        addRecordToDependencies(begin, remaining, returnValue);
      }
      return returnValue;
    }
  }  // namespace eventsetup
}  // namespace edm

#endif
