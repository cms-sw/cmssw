// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataKey
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 31 14:31:13 EST 2005
//

// system include files
#include <memory>
#include <cstring>

// user include files
#include "FWCore/Framework/interface/DataKey.h"

//
// constants, enums and typedefs
//

namespace {
  constexpr char kBlank[] = {'\0'};
}

namespace edm::eventsetup {

  DataKey::DataKey() = default;

  DataKey& DataKey::operator=(const DataKey& rhs) {
    //An exception safe implementation is
    DataKey temp(rhs);
    swap(temp);

    return *this;
  }

  DataKey& DataKey::operator=(DataKey&& rhs) {
    //An exception safe implementation is
    DataKey temp(std::move(rhs));
    swap(temp);

    return *this;
  }

  //
  // member functions
  //
  void DataKey::swap(DataKey& iOther) {
    std::swap(ownMemory_, iOther.ownMemory_);
    // unqualified swap is used for user defined classes.
    // The using directive is needed so that std::swap will be used if there is no other matching swap.
    using std::swap;
    swap(type_, iOther.type_);
    swap(name_, iOther.name_);
  }

  namespace {
    //used for exception safety
    class ArrayHolder {
    public:
      ArrayHolder() = default;

      void swap(ArrayHolder& iOther) {
        const char* t = iOther.ptr_;
        iOther.ptr_ = ptr_;
        ptr_ = t;
      }
      ArrayHolder(const char* iPtr) : ptr_(iPtr) {}
      ~ArrayHolder() { delete[] ptr_; }
      void release() { ptr_ = nullptr; }

    private:
      const char* ptr_{nullptr};
    };
  }  // namespace

  void DataKey::makeCopyOfMemory() {
    //empty string is the most common case, so handle it special

    char* pName = const_cast<char*>(kBlank);
    //NOTE: if in the future additional tags are added then
    // I should make sure that pName gets deleted in the case
    // where an exception is thrown
    ArrayHolder pNameHolder;
    if (kBlank[0] != name().value()[0]) {
      size_t const nBytes = std::strlen(name().value()) + 1;
      pName = new char[nBytes];
      ArrayHolder t(pName);
      pNameHolder.swap(t);
      std::strncpy(pName, name().value(), nBytes);
    }
    name_ = NameTag(pName);
    ownMemory_ = true;
    pNameHolder.release();
  }

  void DataKey::deleteMemory() {
    if (kBlank[0] != name().value()[0]) {
      delete[] const_cast<char*>(name().value());
    }
  }

  //
  // const member functions
  //
  bool DataKey::operator==(const DataKey& iRHS) const { return ((type_ == iRHS.type_) && (name_ == iRHS.name_)); }

  bool DataKey::operator<(const DataKey& iRHS) const {
    return (type_ < iRHS.type_) || ((type_ == iRHS.type_) && (name_ < iRHS.name_));
  }
}  // namespace edm::eventsetup
