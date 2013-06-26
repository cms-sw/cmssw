#ifndef Framework_EventSetupRecordProviderFactoryTemplate_h
#define Framework_EventSetupRecordProviderFactoryTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordProviderFactoryTemplate
// 
/**\class EventSetupRecordProviderFactoryTemplate EventSetupRecordProviderFactoryTemplate.h FWCore/Framework/interface/EventSetupRecordProviderFactoryTemplate.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Mon Mar 28 16:58:15 EST 2005
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordProviderTemplate.h"
#include "FWCore/Framework/interface/EventSetupRecordProviderFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordProviderFactoryManager.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {
   namespace eventsetup {

template<class T>
class EventSetupRecordProviderFactoryTemplate : public EventSetupRecordProviderFactory
{

   public:
      EventSetupRecordProviderFactoryTemplate() {
         EventSetupRecordProviderFactoryManager::instance().addFactory(
               *this,
               EventSetupRecordKey::makeKey<T>());
      }
      //virtual ~EventSetupRecordProviderFactoryTemplate();

      // ---------- const member functions ---------------------
      virtual std::auto_ptr<EventSetupRecordProvider> makeRecordProvider() const {
         return std::auto_ptr<EventSetupRecordProvider>(
                     new EventSetupRecordProviderTemplate<T>());
      }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      EventSetupRecordProviderFactoryTemplate(const EventSetupRecordProviderFactoryTemplate&); // stop default

      const EventSetupRecordProviderFactoryTemplate& operator=(const EventSetupRecordProviderFactoryTemplate&); // stop default

      // ---------- member data --------------------------------

};
   }
}
#endif
