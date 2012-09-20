// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxy
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:49:19 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"


//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {
//
// static data member definitions
//
static
const ComponentDescription*
dummyDescription()
{
   static ComponentDescription s_desc;
   return &s_desc;
}     
//
// constructors and destructor
//
DataProxy::DataProxy() :
   cache_(0),
   cacheIsValid_(false),
   nonTransientAccessRequested_(false),
   description_(dummyDescription())
{
}

// DataProxy::DataProxy(const DataProxy& rhs)
// {
//    // do actual copying here;
// }

DataProxy::~DataProxy()
{
}

//
// assignment operators
//
// const DataProxy& DataProxy::operator=(const DataProxy& rhs)
// {
//   //An exception safe implementation is
//   DataProxy temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
DataProxy::setCacheIsValidAndAccessType(bool iTransientAccessOnly) const { 
   cacheIsValid_ = true;
   if(!iTransientAccessOnly) {
      nonTransientAccessRequested_ = true;
   }
}
      
void DataProxy::clearCacheIsValid() { 
   cacheIsValid_ = false;
   nonTransientAccessRequested_ = false;
   cache_ = 0;
}
      
void 
DataProxy::resetIfTransient() {
   if (!nonTransientAccessRequested_) {
      clearCacheIsValid();
      invalidateTransientCache();
   }
}

void 
DataProxy::invalidateTransientCache() {
   invalidateCache();
}
//
// const member functions
//
namespace  {
   void throwMakeException(const EventSetupRecord& iRecord,
                           const DataKey& iKey)  {
      throw MakeDataException(iRecord.key(),iKey);
   }
}
      
      
const void* 
DataProxy::get(const EventSetupRecord& iRecord, const DataKey& iKey, bool iTransiently) const
{
   if(!cacheIsValid()) {
      cache_ = const_cast<DataProxy*>(this)->getImpl(iRecord, iKey);
   }
   //It is safe to always set cache to valid.
   //We need to set the AccessType for each request so this can't be called in the if block above.
   //This also must be before the cache_ check since we want to setCacheIsValid before a possible
   // exception throw. If we don't, 'getImpl' will be called again on a second request for the data.
   setCacheIsValidAndAccessType(iTransiently);
   if(0 == cache_) {
      throwMakeException(iRecord, iKey);
   }
   return cache_;
}

void DataProxy::doGet(const EventSetupRecord& iRecord, const DataKey& iKey, bool iTransiently) const {
   get(iRecord, iKey, iTransiently);
}
      
      
//
// static member functions
//
   }
}
