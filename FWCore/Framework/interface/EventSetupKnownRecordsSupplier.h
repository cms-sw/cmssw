#ifndef FWCore_Framework_EventSetupKnownRecordsSupplier_h
#define FWCore_Framework_EventSetupKnownRecordsSupplier_h
// -*- C++ -*-
//
// Package:     Framework
// Class:      EventSetupKnownRecordsSupplier
//
/**\class EventSetupKnownRecordsSupplier EventSetupKnownRecordsSupplier.h FWCore/Framework/interface/EventSetupKnownRecordsSupplier.h

 Description: Interface for determining if an EventSetup Record is known to the framework

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 24 14:10:07 EST 2005
//

// user include files

// system include files


// forward declarations
namespace edm {

   namespace eventsetup {
      class EventSetupRecordKey;

class EventSetupKnownRecordsSupplier {

   public:

      EventSetupKnownRecordsSupplier() = default;
      virtual ~EventSetupKnownRecordsSupplier() = default;

      // ---------- const member functions ---------------------
      virtual bool isKnown(EventSetupRecordKey const&) const = 0;

   private:
      EventSetupKnownRecordsSupplier(EventSetupKnownRecordsSupplier const&) = delete;

      EventSetupKnownRecordsSupplier const& operator=(EventSetupKnownRecordsSupplier const&) = delete;

};

   }
}
#endif
