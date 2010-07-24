#ifndef FWCore_Framework_EventSetupRecordImplementation_h
#define FWCore_Framework_EventSetupRecordImplementation_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordImplementation
// 
/**\class EventSetupRecordImplementation EventSetupRecordImplementation.h FWCore/Framework/interface/EventSetupRecordImplementation.h

 Description: Help class which implements the necessary virtual methods for a new Record class

 Usage:
    This class handles implementing the necessary 'meta data' methods for a Record. To use the class, a new Record type should
 inherit from EventSetupRecordImplementation and pass itself as the argument to the template parameter. For example, for a 
 Record named FooRcd, you would declare it like
 
      class FooRcd : public edm::eventsetup::EventSetupRecordImplementation< FooRcd > {};

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 16:50:49 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {
   namespace eventsetup {
     class ComponentDescription;
     
      template<class T>
      class EventSetupRecordImplementation : public EventSetupRecord {
         
      public:
         //virtual ~EventSetupRecordImplementation();
         
         // ---------- const member functions ---------------------
         virtual EventSetupRecordKey key() const {
            return EventSetupRecordKey::makeKey<T>();
         }
         
         // ---------- static member functions --------------------
         static EventSetupRecordKey keyForClass()  {
            return EventSetupRecordKey::makeKey<T>();
         }
         
         // ---------- member functions ---------------------------
         
      protected:
         EventSetupRecordImplementation() {}
         
      private:
         EventSetupRecordImplementation(const EventSetupRecordImplementation&); // stop default
         
         const EventSetupRecordImplementation& operator=(const EventSetupRecordImplementation&); // stop default
         
         // ---------- member data --------------------------------
         
      };
   }
}

#endif
