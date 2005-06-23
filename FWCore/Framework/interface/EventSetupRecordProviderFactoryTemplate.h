#ifndef EVENTSETUP_EVENTSETUPRECORDFACTORYTEMPLATE_H
#define EVENTSETUP_EVENTSETUPRECORDFACTORYTEMPLATE_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     EventSetupRecordProviderFactoryTemplate
// 
/**\class EventSetupRecordProviderFactoryTemplate EventSetupRecordProviderFactoryTemplate.h Core/CoreFramework/interface/EventSetupRecordProviderFactoryTemplate.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Mon Mar 28 16:58:15 EST 2005
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/EventSetupRecordProviderTemplate.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordProviderFactory.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordProviderFactoryManager.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordKey.h"

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
#endif /* EVENTSETUP_EVENTSETUPRECORDFACTORYTEMPLATE_H */
