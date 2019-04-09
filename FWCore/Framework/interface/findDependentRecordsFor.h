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
#include "boost/mpl/begin_end.hpp"
#include "boost/mpl/deref.hpp"
#include "boost/mpl/next.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DependentRecordTag.h"

// forward declarations
namespace edm {
  namespace eventsetup {

    //If the types are the same, stop the recursion
    template <typename T>
    void addRecordToDependencies(const T*, const T*, std::set<EventSetupRecordKey>&) {}

    //Recursively desend container while adding first type's info to set
    template <typename TFirst, typename TEnd>
    void addRecordToDependencies(const TFirst*, const TEnd* iEnd, std::set<EventSetupRecordKey>& oSet) {
      oSet.insert(EventSetupRecordKey::makeKey<typename boost::mpl::deref<TFirst>::type>());
      const typename boost::mpl::next<TFirst>::type* next(nullptr);
      addRecordToDependencies(next, iEnd, oSet);
    }

    //Handle the case where a Record has dependencies
    template <typename T>
    struct FindDependenciesFromDependentRecord {
      inline static void dependentRecords(std::set<EventSetupRecordKey>& oSet) {
        typedef typename T::list_type list_type;
        const typename boost::mpl::begin<list_type>::type* begin(nullptr);
        const typename boost::mpl::end<list_type>::type* end(nullptr);
        addRecordToDependencies(begin, end, oSet);
      }
    };

    //Handle the case where a Record has no dependencies
    struct NoDependenciesForRecord {
      inline static void dependentRecords(std::set<EventSetupRecordKey>&) {}
    };

    template <typename T>
    std::set<EventSetupRecordKey> findDependentRecordsFor() {
      typedef typename boost::mpl::if_<typename std::is_base_of<edm::eventsetup::DependentRecordTag, T>::type,
                                       FindDependenciesFromDependentRecord<T>,
                                       NoDependenciesForRecord>::type DepFinder;
      std::set<EventSetupRecordKey> returnValue;
      DepFinder::dependentRecords(returnValue);
      return returnValue;
    }
  }  // namespace eventsetup
}  // namespace edm

#endif
