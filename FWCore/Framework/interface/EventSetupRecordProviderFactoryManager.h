#ifndef Framework_EventSetupRecordProviderFactoryManager_h
#define Framework_EventSetupRecordProviderFactoryManager_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordProviderFactoryManager
// 
/**\class EventSetupRecordProviderFactoryManager EventSetupRecordProviderFactoryManager.h FWCore/Framework/interface/EventSetupRecordProviderFactoryManager.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Mon Mar 28 16:58:30 EST 2005
//

// system include files
#include <map>
#include <memory>
// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupRecordProviderFactory;
      class EventSetupRecordProvider;
      
class EventSetupRecordProviderFactoryManager
{
   friend class NonExistentClassToSuppressStupidCompilerWarning;

   public:
      virtual ~EventSetupRecordProviderFactoryManager();

      // ---------- const member functions ---------------------
      std::auto_ptr<EventSetupRecordProvider> makeRecordProvider(const EventSetupRecordKey&) const;

      // ---------- static member functions --------------------
      static EventSetupRecordProviderFactoryManager& instance();
   
      // ---------- member functions ---------------------------
      void addFactory(const EventSetupRecordProviderFactory&, 
                       const EventSetupRecordKey&);

   private:
      EventSetupRecordProviderFactoryManager();
      EventSetupRecordProviderFactoryManager(const EventSetupRecordProviderFactoryManager&); // stop default

      const EventSetupRecordProviderFactoryManager& operator=(const EventSetupRecordProviderFactoryManager&); // stop default

      // ---------- member data --------------------------------
      std::map<EventSetupRecordKey, const EventSetupRecordProviderFactory*> factories_;
};

   }
}
#endif
