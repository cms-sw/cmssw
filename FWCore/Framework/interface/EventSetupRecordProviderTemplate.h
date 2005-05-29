#ifndef EVENTSETUP_EVENTSETUPRECORDPROVIDERTEMPLATE_H
#define EVENTSETUP_EVENTSETUPRECORDPROVIDERTEMPLATE_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     EventSetupRecordProviderTemplate
// 
/**\class EventSetupRecordProviderTemplate EventSetupRecordProviderTemplate.h Core/CoreFramework/interface/EventSetupRecordProviderTemplate.h

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
#include "boost/mpl/if.hpp"
#include "boost/type_traits/is_base_and_derived.hpp"
// user include files
#include "FWCore/CoreFramework/interface/EventSetupRecordProvider.h"
#include "FWCore/CoreFramework/interface/DependentEventSetupRecordProviderTemplate.h"
#include "FWCore/CoreFramework/interface/DependentRecordImplementation.h"
#include "FWCore/CoreFramework/interface/EventSetupProvider.h"
#include "FWCore/CoreFramework/interface/DataProxyProvider.h"

// forward declarations
namespace edm {
   namespace eventsetup {

template<class T>
      class EventSetupRecordProviderTemplate : 
      public  boost::mpl::if_< typename boost::is_base_and_derived<edm::eventsetup::DependentRecordTag, T>::type,
                             DependentEventSetupRecordProviderTemplate<T>,
      EventSetupRecordProvider>::type
{

   public:
      typedef T RecordType;
   typedef  typename boost::mpl::if_< typename boost::is_base_and_derived<edm::eventsetup::DependentRecordTag, T>::type,
      DependentEventSetupRecordProviderTemplate<T>,
      EventSetupRecordProvider >::type    BaseType;
   
   EventSetupRecordProviderTemplate() : BaseType(EventSetupRecordKey::makeKey<T>() ) {}
      //virtual ~EventSetupRecordProviderTemplate();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   virtual void addRecordTo(EventSetupProvider& iEventSetupProvider ) {
      record_.set( this->validityInterval() );
      iEventSetupProvider.addRecordToEventSetup( record_ );
   }
   
protected:
      virtual void addProxiesToRecord( boost::shared_ptr<DataProxyProvider> iProvider) {
         typedef DataProxyProvider::KeyedProxies ProxyList ;

         const ProxyList& keyedProxies( iProvider->keyedProxies( key() ) ) ;
         ProxyList::const_iterator finishedProxyList( keyedProxies.end() ) ;
         for ( ProxyList::const_iterator keyedProxy( keyedProxies.begin() ) ;
               keyedProxy != finishedProxyList ;
               ++keyedProxy ) {
            record_.add( (*keyedProxy).first , (*keyedProxy).second.get() ) ;
         }
      }
   private:
      EventSetupRecordProviderTemplate( const EventSetupRecordProviderTemplate& ); // stop default

      const EventSetupRecordProviderTemplate& operator=( const EventSetupRecordProviderTemplate& ); // stop default

      // ---------- member data --------------------------------
      T record_;
};

   }
}
#endif /* EVENTSETUP_EVENTSETUPRECORDPROVIDERTEMPLATE_H */
