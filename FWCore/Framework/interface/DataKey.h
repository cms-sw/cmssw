#ifndef Framework_DataKey_h
#define Framework_DataKey_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataKey
//
/**\class DataKey DataKey.h FWCore/Framework/interface/DataKey.h

 Description: Key used to identify data within a EventSetupRecord

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 31 14:31:03 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/DataKeyTags.h"
#include "FWCore/Framework/interface/HCTypeTag.h"

// forward declarations
namespace edm::eventsetup {
  class DataKey
  {

    friend void swap(DataKey&, DataKey&);
  public:
    enum DoNotCopyMemory { kDoNotCopyMemory };

    DataKey();
    DataKey(const TypeTag& iType,
            const IdTags& iId) :
      type_(iType),
      name_(iId),
      ownMemory_() {
      makeCopyOfMemory();
    }

    DataKey(const TypeTag& iType,
            const IdTags& iId,
            DoNotCopyMemory) :
      type_(iType),
      name_(iId),
      ownMemory_(false) {}

    DataKey(const DataKey& iRHS) :
      type_(iRHS.type_),
      name_(iRHS.name_),
      ownMemory_() {
      makeCopyOfMemory();
    }

    DataKey& operator=(const DataKey&);

    ~DataKey() { releaseMemory(); }

    // ---------- const member functions ---------------------
    const TypeTag& type() const { return type_; }
    const NameTag& name() const { return name_; }

    bool operator==(const DataKey& iRHS) const;
    bool operator<(const DataKey& iRHS) const;

    // ---------- static member functions --------------------
    template<class T>
    static TypeTag makeTypeTag() {
      return heterocontainer::HCTypeTag::make<T>();
    }

    // ---------- member functions ---------------------------

  private:
    void makeCopyOfMemory();
    void releaseMemory() {
      if(ownMemory_) {
        deleteMemory();
        ownMemory_ = false;
      }
    }
    void deleteMemory();
    void swap(DataKey&);

    // ---------- member data --------------------------------
    TypeTag type_{};
    NameTag name_{};
    bool ownMemory_{false};
  };

  // Free swap function
  inline
  void
  swap(DataKey& a, DataKey& b)
  {
    a.swap(b);
  }
}
#endif
