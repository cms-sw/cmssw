#ifndef Framework_DependentEventSetupRecordProviderTemplate_h
#define Framework_DependentEventSetupRecordProviderTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DependentEventSetupRecordProviderTemplate
// 
/**\class DependentEventSetupRecordProviderTemplate DependentEventSetupRecordProviderTemplate.h FWCore/Framework/interface/DependentEventSetupRecordProviderTemplate.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Sun May  1 16:59:49 EDT 2005
// $Id: DependentEventSetupRecordProviderTemplate.h,v 1.5 2005/09/01 23:30:48 wmtan Exp $
//

// system include files
#include "boost/mpl/begin_end.hpp"
#include "boost/mpl/deref.hpp"
#include "boost/mpl/next.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"

// forward declarations
namespace edm {
   namespace eventsetup {
template <class T>
class DependentEventSetupRecordProviderTemplate : public EventSetupRecordProvider
{

   public:
      typedef typename  T::list_type list_type;

      DependentEventSetupRecordProviderTemplate(const EventSetupRecordKey& iKey) :
      EventSetupRecordProvider(iKey) {}
   //virtual ~DependentEventSetupRecordProviderTemplate();

      // ---------- const member functions ---------------------
      std::set<EventSetupRecordKey> dependentRecords() const {
         std::set<EventSetupRecordKey> returnValue;
         const  typename boost::mpl::begin<list_type>::type * begin(0);
         const  typename boost::mpl::end<list_type>::type * end(0);
         addRecordToDependencies(begin, end, returnValue);
         return returnValue;
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      template< class TFirst, class TEnd>
      void addRecordToDependencies(const TFirst*, const TEnd* iEnd, 
                                    std::set<EventSetupRecordKey>& oSet) const {
         oSet.insert(EventSetupRecordKey::makeKey<typename boost::mpl::deref<TFirst>::type>());
         const  typename boost::mpl::next< TFirst >::type * next(0);
         addRecordToDependencies(next, iEnd, oSet);
      }
      
         void addRecordToDependencies(const typename boost::mpl::end<list_type>::type*,
                                       const typename boost::mpl::end<list_type>::type*,
                                       std::set<EventSetupRecordKey>&) const { }
         
      DependentEventSetupRecordProviderTemplate(const DependentEventSetupRecordProviderTemplate&); // stop default

      const DependentEventSetupRecordProviderTemplate& operator=(const DependentEventSetupRecordProviderTemplate&); // stop default

      // ---------- member data --------------------------------

};

}
   }
#endif
