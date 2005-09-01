#ifndef Framework_EventSetupRecordProviderFactory_h
#define Framework_EventSetupRecordProviderFactory_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordProviderFactory
// 
/**\class EventSetupRecordProviderFactory EventSetupRecordProviderFactory.h FWCore/Framework/interface/EventSetupRecordProviderFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Mon Mar 28 16:58:12 EST 2005
//

// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"

// forward declarations
namespace edm {
   namespace eventsetup {
class EventSetupRecordProviderFactory
{

   public:
      virtual ~EventSetupRecordProviderFactory();

      // ---------- const member functions ---------------------
      virtual std::auto_ptr<EventSetupRecordProvider> makeRecordProvider() const = 0;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   protected:
      EventSetupRecordProviderFactory() {}
   private:
      EventSetupRecordProviderFactory(const EventSetupRecordProviderFactory&); // stop default

      const EventSetupRecordProviderFactory& operator=(const EventSetupRecordProviderFactory&); // stop default

      // ---------- member data --------------------------------

};
   }
}

#endif
