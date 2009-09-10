#ifndef FWCore_Framework_EventSetupRecordImplementation_h
#define FWCore_Framework_EventSetupRecordImplementation_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordImplementation
// 
/**\class EventSetupRecordImplementation EventSetupRecordImplementation.h FWCore/Framework/interface/EventSetupRecordImplementation.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 16:50:49 EST 2005
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

// forward declarations
namespace edm {
   namespace eventsetup {
     class ComponentDescription;
     
template<class T>
class EventSetupRecordImplementation : public EventSetupRecord
{

   public:
      //virtual ~EventSetupRecordImplementation();

      // ---------- const member functions ---------------------
      template< typename HolderT>
         void get(HolderT& iHolder) const {
            const typename HolderT::value_type* value = 0;
            const ComponentDescription* desc = 0;
            this->getImplementation(value, "",desc);
                                                      
            iHolder = HolderT(value,desc);
         }

      template< typename HolderT>
      void get(const char* iName, HolderT& iHolder) const {
         const typename HolderT::value_type* value = 0;
         const ComponentDescription* desc = 0;
         this->getImplementation(value, iName,desc);
         iHolder = HolderT(value,desc);
      }
      template< typename HolderT>
      void get(const std::string& iName, HolderT& iHolder) const {
         const typename HolderT::value_type* value = 0;
         const ComponentDescription* desc = 0;
         this->getImplementation(value, iName.c_str(),desc);
         iHolder = HolderT(value,desc);
      }
      
      template< typename HolderT>
      void get(const edm::ESInputTag& iTag, HolderT& iHolder) const {
         const typename HolderT::value_type* value = 0;
         const ComponentDescription* desc = 0;
         this->getImplementation(value, iTag.data().c_str(),desc);
         validate(desc,iTag);
         iHolder = HolderT(value,desc);
      }
   
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
      
      template < typename DataT > 
         void getImplementation(DataT const *& iData ,
                                const char* iName,
                                const ComponentDescription*& iDesc) const;       // ---------- member data --------------------------------

};
   }
}
#include "FWCore/Framework/interface/recordGetImplementation.icc"

#endif
