#ifndef Framework_EventSetupRecordKey_h
#define Framework_EventSetupRecordKey_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordKey
// 
/**\class EventSetupRecordKey EventSetupRecordKey.h FWCore/Framework/interface/EventSetupRecordKey.h

 Description: Key used to lookup a EventSetupRecord within the EventSetup

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Mar 25 15:19:21 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/HCTypeTag.h"
#include "FWCore/Framework/interface/HCMethods.h"

// forward declarations
namespace edm {
   namespace eventsetup {
class EventSetupRecordKey
{

   public:
   typedef heterocontainer::HCTypeTag TypeTag;
      
      EventSetupRecordKey();
      EventSetupRecordKey(const TypeTag& iType) :
         type_(iType) {}

      //virtual ~EventSetupRecordKey();

      // ---------- const member functions ---------------------
      const TypeTag& type() const { return type_;}
      
      bool operator<(const EventSetupRecordKey& iRHS) const {
         return type_ < iRHS.type_;
      }
      bool operator==(const EventSetupRecordKey& iRHS) const {
         return type_ == iRHS.type_;
      }
      
      const char* name() const { return type().name(); }
      // ---------- static member functions --------------------
      template<class T>
         static EventSetupRecordKey makeKey() {
            return eventsetup::heterocontainer::makeKey<T,EventSetupRecordKey>();
         }
      
      // ---------- member functions ---------------------------

   private:
      //EventSetupRecordKey(const EventSetupRecordKey&); // allow default

      //const EventSetupRecordKey& operator=(const EventSetupRecordKey&); // allow default

      // ---------- member data --------------------------------
      TypeTag type_;
         
};
   }
}

#endif
