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
#include "boost/shared_ptr.hpp"
#include "boost/type_traits/is_base_and_derived.hpp"
// user include files
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/DependentEventSetupRecordProviderTemplate.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"

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
   
   EventSetupRecordProviderTemplate() : BaseType(EventSetupRecordKey::makeKey<T>()), record_() {}
      //virtual ~EventSetupRecordProviderTemplate();

      // ---------- const member functions ---------------------
      const T& record() const {return record_;}
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   virtual void addRecordTo(EventSetupProvider& iEventSetupProvider) {
      record_.set(this->validityInterval());
      iEventSetupProvider.addRecordToEventSetup(record_);
   }
   
protected:
      virtual void addProxiesToRecord(boost::shared_ptr<DataProxyProvider> iProvider,
                                      const EventSetupRecordProvider::DataToPreferredProviderMap& iMap) {
         typedef DataProxyProvider::KeyedProxies ProxyList ;
         typedef EventSetupRecordProvider::DataToPreferredProviderMap PreferredMap;

         const ProxyList& keyedProxies(iProvider->keyedProxies(this->key())) ;
         ProxyList::const_iterator finishedProxyList(keyedProxies.end()) ;
         for (ProxyList::const_iterator keyedProxy(keyedProxies.begin()) ;
               keyedProxy != finishedProxyList ;
               ++keyedProxy) {
            PreferredMap::const_iterator itFound = iMap.find(keyedProxy->first);
            if(iMap.end() != itFound) {
               if( itFound->second.type_ != keyedProxy->second->providerDescription()->type_ ||
                   itFound->second.label_ != keyedProxy->second->providerDescription()->label_ ) {
                  //this is not the preferred provider
                  continue;
               }
            }
            record_.add((*keyedProxy).first , (*keyedProxy).second.get()) ;
         }
      }
   private:
      virtual void cacheReset() {
        record_.cacheReset();
      }
      virtual bool checkResetTransients() {
         return record_.transientReset();
      }
      EventSetupRecordProviderTemplate(const EventSetupRecordProviderTemplate&); // stop default

      const EventSetupRecordProviderTemplate& operator=(const EventSetupRecordProviderTemplate&); // stop default

      // ---------- member data --------------------------------
      T record_;
};

   }
}
#endif
