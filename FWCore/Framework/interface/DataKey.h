#ifndef EVENTSETUP_DATAKEY_H
#define EVENTSETUP_DATAKEY_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DataKey
// 
/**\class DataKey DataKey.h Core/CoreFramework/interface/DataKey.h

 Description: Key used to identify data within a EventSetupRecord

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 31 14:31:03 EST 2005
// $Id: DataKey.h,v 1.2 2005/04/04 20:38:02 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/DataKeyTags.h"
#include "FWCore/CoreFramework/interface/HCTypeTagTemplate.h"

// forward declarations
namespace edm {
   namespace eventsetup {
class DataKey
{

   public:
   enum DoNotCopyMemory { kDoNotCopyMemory };
   
      DataKey();
      DataKey( const TypeTag& iType, 
               const IdTags& iId ) :
         type_(iType),
         name_(iId) { makeCopyOfMemory();}

      DataKey( const TypeTag& iType, 
               const IdTags& iId,
               DoNotCopyMemory ) :
         type_(iType),
         name_(iId),
         ownMemory_(false) {}
      
      DataKey( const DataKey& iRHS) : 
         type_(iRHS.type_),
         name_(iRHS.name_) {
            makeCopyOfMemory();
         }
      
      const DataKey& operator=( const DataKey& ); // stop default
      
      ~DataKey() { releaseMemory(); }
      
      // ---------- const member functions ---------------------
      const TypeTag& type() const { return type_; }
      const NameTag& name() const { return name_; }
      
      bool operator==( const DataKey& iRHS ) const;
      bool operator<( const DataKey& iRHS) const;
      
      // ---------- static member functions --------------------
      template<class T>
         static TypeTag makeTypeTag() {
            return heterocontainer::HCTypeTagTemplate<T, DataKey>();
         }
      
      // ---------- member functions ---------------------------

   private:
      void makeCopyOfMemory();
      void releaseMemory() {
         if( ownMemory_ ) {
            deleteMemory();
            ownMemory_ = false;
         }
      }
      void deleteMemory();
      void swap(DataKey& );
      
      // ---------- member data --------------------------------
      TypeTag type_;
      NameTag name_;
      bool ownMemory_;
};

   }
}
#endif /* EVENTSETUP_DATAKEY_H */
