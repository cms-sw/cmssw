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
#include <mutex>

// user include files
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {
     static std::recursive_mutex s_esGlobalMutex;
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
   cache_(nullptr),
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
void DataProxy::clearCacheIsValid() {
   cacheIsValid_.store(false, std::memory_order_release);
   nonTransientAccessRequested_.store(false, std::memory_order_release);
   cache_ = nullptr;
}
      
void 
DataProxy::resetIfTransient() {
   if (!nonTransientAccessRequested_.load(std::memory_order_acquire)) {
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
   void throwMakeException(const EventSetupRecordImpl& iRecord,
                           const DataKey& iKey)  {
      throw MakeDataException(iRecord.key(),iKey);
   }

   class ESSignalSentry {
   public:
      ESSignalSentry(const EventSetupRecordImpl& iRecord,
                     const DataKey& iKey,
                     ComponentDescription const* componentDescription,
                     ActivityRegistry const* activityRegistry) :
         eventSetupRecord_(iRecord),
         dataKey_(iKey),
         componentDescription_(componentDescription),
         calledPostLock_(false),
         activityRegistry_(activityRegistry) {

         activityRegistry->preLockEventSetupGetSignal_(componentDescription_, eventSetupRecord_.key(), dataKey_);
      }
      void sendPostLockSignal() {
         calledPostLock_ = true;
         activityRegistry_->postLockEventSetupGetSignal_(componentDescription_, eventSetupRecord_.key(), dataKey_);
      }
      ~ESSignalSentry() noexcept(false) {
         if (!calledPostLock_) {
            activityRegistry_->postLockEventSetupGetSignal_(componentDescription_, eventSetupRecord_.key(), dataKey_);
         }
         activityRegistry_->postEventSetupGetSignal_(componentDescription_, eventSetupRecord_.key(), dataKey_);
      }
   private:
      EventSetupRecordImpl const& eventSetupRecord_;
      DataKey const& dataKey_;
      ComponentDescription const* componentDescription_;
      bool calledPostLock_;
      ActivityRegistry const* activityRegistry_;
   };
}

const void* 
DataProxy::get(const EventSetupRecordImpl& iRecord, const DataKey& iKey, bool iTransiently, ActivityRegistry const* activityRegistry) const
{
   if(!cacheIsValid()) {
      ESSignalSentry signalSentry(iRecord, iKey, providerDescription(), activityRegistry);
      std::lock_guard<std::recursive_mutex> guard(s_esGlobalMutex);
      signalSentry.sendPostLockSignal();
      if(!cacheIsValid()) {
         cache_ = const_cast<DataProxy*>(this)->getImpl(iRecord, iKey);
         cacheIsValid_.store(true,std::memory_order_release);
      }
   }
   //We need to set the AccessType for each request so this can't be called in the if block above.
   //This also must be before the cache_ check since we want to setCacheIsValid before a possible
   // exception throw. If we don't, 'getImpl' will be called again on a second request for the data.
   if(!iTransiently) {
      nonTransientAccessRequested_.store(true, std::memory_order_release);
   }

   if(nullptr == cache_) {
      throwMakeException(iRecord, iKey);
   }
   return cache_;
}

void DataProxy::doGet(const EventSetupRecordImpl& iRecord, const DataKey& iKey, bool iTransiently, ActivityRegistry const* activityRegistry) const {
   get(iRecord, iKey, iTransiently, activityRegistry);
}
      
      
//
// static member functions
//
   }
}
