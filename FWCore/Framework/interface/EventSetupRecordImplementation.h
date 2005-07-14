#ifndef EVENTSETUP_EVENTSETUPRECORDIMPLEMENTATION_H
#define EVENTSETUP_EVENTSETUPRECORDIMPLEMENTATION_H
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
// $Id: EventSetupRecordImplementation.h,v 1.4 2005/06/28 14:39:41 chrjones Exp $
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
//#include "FWCore/Framework/interface/DataKey.h"
//#include "FWCore/Framework/interface/DataProxyTemplate.h"
//#include "FWCore/Framework/interface/NoProxyException.h"

// forward declarations
namespace edm {
   namespace eventsetup {

template<class T>
class EventSetupRecordImplementation : public EventSetupRecord
{

   public:
      //virtual ~EventSetupRecordImplementation();

      // ---------- const member functions ---------------------
      template< typename HolderT>
         void get(HolderT& iHolder, const char* iName = "") const {
            const typename HolderT::value_type* value;
            this->getImplementation(value, iName);
            iHolder = HolderT(value);
         }

   template< typename HolderT>
   void get(HolderT& iHolder, const std::string& iName) const {
      const typename HolderT::value_type* value;
      this->getImplementation(value, iName.c_str());
      iHolder = HolderT(value);
   }
   
   virtual EventSetupRecordKey key() const {
      return EventSetupRecordKey::makeKey<T>();
   }

   virtual bool doGet(const DataKey& aKey) const {
      const DataProxy* proxy = find(aKey);
      if( 0 != proxy ) {
         proxy->doGet( *this, aKey );
      }
      return 0 != proxy;
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
                                const char* iName) const; /* {
            DataKey key(DataKey::makeTypeTag<DataT>(),
                        iName,
                        DataKey::kDoNotCopyMemory);
            
            const DataProxyTemplate<T, DataT>* proxy = 
               static_cast<const DataProxyTemplate<T, DataT>* >(this->find(key));
            
            const DataT* hold = 0;
            if(0 != proxy) {
               // add data key to the stack
               //DAExceptionStackEntry stackEntry(d_key);
               
               hold = proxy->get(static_cast<const T&>(*this), key);
            } else {
               // add durable data key to the stack in order to catch it
               //DAExceptionStackEntry stackEntry(d_key,DAExceptionStackEntry::kUseDurable)
               ;
               
               throw NoProxyException<DataT>(*this, key);
            }
            iData = hold;
         }
      */
      // ---------- member data --------------------------------

};
/*
template <typename T>
template < typename DataT > 
void 
EventSetupRecordImplementation<T>::getImplementation(DataT const *& iData ,
                                                     const char* iName) const 
{
   DataKey key(DataKey::makeTypeTag<DataT>(),
               iName,
               DataKey::kDoNotCopyMemory);
   
   const DataProxyTemplate<T, DataT>* proxy = 
      static_cast<const DataProxyTemplate<T, DataT>* >(this->find(key));
   
   const DataT* hold = 0;
   if(0 != proxy) {
      // add data key to the stack
      //DAExceptionStackEntry stackEntry(d_key);
      
      hold = proxy->get(static_cast<const T&>(*this), key);
   } else {
      // add durable data key to the stack in order to catch it
      //DAExceptionStackEntry stackEntry(d_key,DAExceptionStackEntry::kUseDurable)
      ;
      
      throw NoProxyException<DataT>(*this, key);
   }
   iData = hold;
}
*/
   }
}
#include "FWCore/Framework/interface/recordGetImplementation.icc"

#endif /* EVENTSETUP_EVENTSETUPRECORDIMPLEMENTATION_H */
