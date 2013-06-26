#ifndef FWCore_Framework_EventSetupRecordProviderTemplate_h
#define FWCore_Framework_EventSetupRecordProviderTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordProviderTemplate
// 
/**\class EventSetupRecordProviderTemplate EventSetupRecordProviderTemplate.h FWCore/Framework/interface/EventSetupRecordProviderTemplate.h

 Description: <one line class summary>

 Usage:
    NOTE: The class inherits from DependentEventSetupRecordProvider only if T inherits from DependentRecordTag, else the class inherits directly
           from EventSetupRecordProvider.

*/
//
// Author:      Chris Jones
// Created:     Mon Mar 28 11:43:05 EST 2005
//

// system include files
#include "boost/type_traits/is_base_and_derived.hpp"
#include "boost/mpl/begin_end.hpp"
#include "boost/mpl/deref.hpp"
#include "boost/mpl/next.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/DependentRecordTag.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      
      //If the types are the same, stop the recursion
      template <typename T>
      void addRecordToDependencies(const T*,
                                   const T*,
                                   std::set<EventSetupRecordKey>&) { }
      
      //Recursively desend container while adding first type's info to set
      template< typename TFirst, typename TEnd>
      void addRecordToDependencies(const TFirst*, const TEnd* iEnd, 
                                   std::set<EventSetupRecordKey>& oSet) {
         oSet.insert(EventSetupRecordKey::makeKey<typename boost::mpl::deref<TFirst>::type>());
         const  typename boost::mpl::next< TFirst >::type * next(nullptr);
         addRecordToDependencies(next, iEnd, oSet);
      }

      //Handle the case where a Record has dependencies
      template <typename T>
      struct FindDependenciesFromDependentRecord {
         inline static void dependentRecords(std::set<EventSetupRecordKey>& oSet)  {
            typedef typename T::list_type list_type;
            const  typename boost::mpl::begin<list_type>::type * begin(nullptr);
            const  typename boost::mpl::end<list_type>::type * end(nullptr);
            addRecordToDependencies(begin, end, oSet);
         }
      };
      
      //Handle the case where a Record has no dependencies
      struct NoDependenciesForRecord {
         inline static void dependentRecords(std::set<EventSetupRecordKey>&)  {
         }
      };
      
      template <typename T>
      std::set<EventSetupRecordKey>
      findDependentRecordsFor() {
         typedef typename boost::mpl::if_< typename boost::is_base_and_derived<edm::eventsetup::DependentRecordTag, T>::type,
                                           FindDependenciesFromDependentRecord<T>,
                                           NoDependenciesForRecord>::type DepFinder;
         std::set<EventSetupRecordKey> returnValue;
         DepFinder::dependentRecords(returnValue);
         return returnValue;
      }
      

      template<class T>
      class EventSetupRecordProviderTemplate : public EventSetupRecordProvider
      {
         
      public:
         typedef T RecordType;
         typedef EventSetupRecordProvider    BaseType;
         
         EventSetupRecordProviderTemplate() : BaseType(EventSetupRecordKey::makeKey<T>()), record_() {}
         //virtual ~EventSetupRecordProviderTemplate();
         
         // ---------- const member functions ---------------------
         EventSetupRecord const& record() const {return record_;}
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         std::set<EventSetupRecordKey> dependentRecords() const {
            return findDependentRecordsFor<T>();
         }
      protected:
         EventSetupRecord& record() { return record_; }
         
      private:
         EventSetupRecordProviderTemplate(EventSetupRecordProviderTemplate const&); // stop default
         
         EventSetupRecordProviderTemplate const& operator=(EventSetupRecordProviderTemplate const&); // stop default
         
         // ---------- member data --------------------------------
         T record_;
      };
      
   }
}
#endif
