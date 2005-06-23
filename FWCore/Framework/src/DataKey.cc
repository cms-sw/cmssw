// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DataKey
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 31 14:31:13 EST 2005
// $Id: DataKey.cc,v 1.1 2005/05/29 02:29:53 wmtan Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/CoreFramework/interface/DataKey.h"


//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {
//
// static data member definitions
//

//
// constructors and destructor
//
DataKey::DataKey(): ownMemory_(false)
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
   std::swap(type_, iOther.type_);
   std::swap(name_, iOther.name_);
}

void 
DataKey::makeCopyOfMemory()
{
   //empty string is the most common case, so handle it special
   static const char kBlank = '\0';
   
   char* pName = const_cast<char*>(&kBlank);
   //NOTE: if in the future additional tags are added then 
   // I should make sure that pName gets deleted in the case
   // where an exception is thrown
   std::auto_ptr<char> pNameHolder;
   if(kBlank != name().value()[0]) {
      pName = new char[ std::strlen(name().value()) + 1];
      pNameHolder = std::auto_ptr<char>(pName);
      std::strcpy(pName, name().value());
   }
   name_ = NameTag(pName);
   ownMemory_ = true;
   pNameHolder.release();
}

void
DataKey::deleteMemory()
{
   static const char kBlank = '\0';
   
   if(kBlank != name().value()[0]) {
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

#include "FWCore/CoreFramework/interface/HCTypeTag.icc"
template class edm::eventsetup::heterocontainer::HCTypeTag<edm::eventsetup::DataKey>;
