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

static const char kBlank[] = {'\0'};

namespace edm {
   namespace eventsetup {
//
// static data member definitions
//

//
// constructors and destructor
//
DataKey::DataKey(): type_(), name_(), ownMemory_(false)
{
}

// DataKey::DataKey(const DataKey& rhs)
// {
//    // do actual copying here;
// }

//DataKey::~DataKey()
//{
//}

//
// assignment operators
//
const DataKey& DataKey::operator=(const DataKey& rhs)
{
   //An exception safe implementation is
   DataKey temp(rhs);
   swap(temp);

   return *this;
}

//
// member functions
//
void
DataKey::swap(DataKey& iOther)
{
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
            ArrayHolder():ptr_(0){}
            
            void swap(ArrayHolder& iOther) {
               const char* t = iOther.ptr_;
               iOther.ptr_ = ptr_;
               ptr_ = t;
            }
            ArrayHolder(const char* iPtr): ptr_(iPtr) {}
            ~ArrayHolder() { delete [] ptr_; }
            void release() { ptr_=0;}
         private:
            const char* ptr_;
         };
      }
void 
DataKey::makeCopyOfMemory()
{
   //empty string is the most common case, so handle it special
   
   char* pName = const_cast<char*>(kBlank);
   //NOTE: if in the future additional tags are added then 
   // I should make sure that pName gets deleted in the case
   // where an exception is thrown
   ArrayHolder pNameHolder;
   if(kBlank[0] != name().value()[0]) {
      pName = new char[ std::strlen(name().value()) + 1];
      ArrayHolder t(pName);
      pNameHolder.swap(t);
      std::strcpy(pName, name().value());
   }
   name_ = NameTag(pName);
   ownMemory_ = true;
   pNameHolder.release();
}

void
DataKey::deleteMemory()
{
   if(kBlank[0] != name().value()[0]) {
      delete [] const_cast<char*>(name().value());
   }
}

//
// const member functions
//
bool
DataKey::operator==(const DataKey& iRHS) const 
{
   return ((type_ == iRHS.type_) &&
            (name_ == iRHS.name_));
}

bool
DataKey::operator<(const DataKey& iRHS) const 
{
   return (type_ < iRHS.type_) ||
   ((type_ == iRHS.type_) && (name_ < iRHS.name_));
/*
   if(type_ < iRHS.type_) {
      return true;
   } else if (type_ == iRHS.type_) {
      if(name_ < iRHS.name_) {
         return true;
   }
   return false;
      */
}

//
// static member functions
//
   }
}
