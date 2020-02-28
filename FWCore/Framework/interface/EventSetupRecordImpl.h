#ifndef FWCore_Framework_EventSetupRecordImpl_h
#define FWCore_Framework_EventSetupRecordImpl_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordImpl
//
/**\class edm::eventsetup::EventSetupRecordImpl

 Description: Base class for all Records in an EventSetup.  Holds data with the same lifetime.

 Usage:
This class contains the Proxies that make up a given Record.  It
is designed to be reused time after time, rather than it being
destroyed and a new one created every time a new Record is
required.  Proxies can only be added by the EventSetupRecordProvider class which
uses the 'add' function to do this.

When the set of  Proxies for a Records changes, i.e. a
DataProxyProvider is added of removed from the system, then the
Proxies in a Record need to be changed as appropriate.
In this design it was decided the easiest way to achieve this was
to erase all Proxies in a Record.

It is important for the management of the Records that each Record
know the ValidityInterval that represents the time over which its data is valid.
The ValidityInterval is set by its EventSetupRecordProvider using the
'set' function.  This quantity can be recovered
through the 'validityInterval' method.

*/
//
// Author:      Chris Jones
// Created:     Fri Mar 25 14:38:35 EST 2005
//

// user include files
#include "FWCore/Framework/interface/FunctorESHandleExceptionFactory.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/NoProxyException.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/ESIndices.h"

// system include files
#include <exception>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <atomic>
#include <cassert>

// forward declarations
namespace cms {
  class Exception;
}

namespace edm {

  class ActivityRegistry;
  class ESHandleExceptionFactory;
  class ESInputTag;
  class EventSetupImpl;

  namespace eventsetup {
    struct ComponentDescription;
    class DataProxy;

    class EventSetupRecordImpl {
      friend class EventSetupRecord;

    public:
      EventSetupRecordImpl(const EventSetupRecordKey& iKey, ActivityRegistry const*, unsigned int iovIndex = 0);
      EventSetupRecordImpl(EventSetupRecordImpl const&) = delete;
      EventSetupRecordImpl const& operator=(EventSetupRecordImpl const&) = delete;
      EventSetupRecordImpl(EventSetupRecordImpl&&);
      EventSetupRecordImpl& operator=(EventSetupRecordImpl&&);

      ValidityInterval validityInterval() const;

      ///returns false if no data available for key
      bool doGet(DataKey const& aKey, EventSetupImpl const*, bool aGetTransiently = false) const;

      /**returns true only if someone has already requested data for this key
          and the data was retrieved
          */
      bool wasGotten(DataKey const& aKey) const;

      /**returns the ComponentDescription for the module which creates the data or 0
          if no module has been registered for the data. This does not cause the data to
          actually be constructed.
          */
      ComponentDescription const* providerDescription(DataKey const& aKey) const;

      EventSetupRecordKey const& key() const { return key_; }

      /**If you are caching data from the Record, you should also keep
          this number.  If this number changes then you know that
          the data you have cached is invalid. This is NOT true if
          if the validityInterval() hasn't changed since it is possible that
          the job has gone to a new Record and then come back to the
          previous SyncValue and your algorithm didn't see the intervening
          Record.
          The value of '0' will never be returned so you can use that to
          denote that you have not yet checked the value.
          */
      unsigned long long cacheIdentifier() const { return cacheIdentifier_; }

      unsigned int iovIndex() const { return iovIndex_; }

      ///clears the oToFill vector and then fills it with the keys for all registered data keys
      void fillRegisteredDataKeys(std::vector<DataKey>& oToFill) const;
      ///there is a 1-to-1 correspondence between elements returned and the elements returned from fillRegisteredDataKey.
      std::vector<ComponentDescription const*> componentsForRegisteredDataKeys() const;

      // The following member functions should only be used by EventSetupRecordProvider
      bool add(DataKey const& iKey, DataProxy* iProxy);
      void clearProxies();

      ///Set the cache identifier and validity interval when starting a new IOV
      ///In addition, also notify the DataProxy's a new IOV is starting.
      ///(As a performance optimization, we only notify the DataProxy's if hasFinder
      ///is true. At the current time, the CondDBESSource DataProxy's are the only
      ///ones who need to know about this and they always have finders).
      void initializeForNewIOV(unsigned long long iCacheIdentifier, ValidityInterval const&, bool hasFinder);

      /**Set the validity interval in a thread safe way. This is used when the
         IOV is already in use and the end of the IOV needs to be updated but
         the start time stays the same. In this case a new IOV does not need
         to be started.
          */
      void setSafely(ValidityInterval const&) const;

      void getESProducers(std::vector<ComponentDescription const*>& esproducers) const;

      DataProxy const* find(DataKey const& aKey) const;

      void validate(ComponentDescription const*, ESInputTag const&) const;

      void addTraceInfoToCmsException(cms::Exception& iException,
                                      char const* iName,
                                      ComponentDescription const*,
                                      DataKey const&) const;

      void invalidateProxies();
      void resetIfTransientInProxies();

    private:
      void const* getFromProxy(DataKey const& iKey,
                               ComponentDescription const*& iDesc,
                               bool iTransientAccessOnly,
                               EventSetupImpl const* = nullptr) const;

      void const* getFromProxy(ESProxyIndex iProxyIndex,
                               bool iTransientAccessOnly,
                               ComponentDescription const*& iDesc,
                               DataKey const*& oGottenKey,
                               EventSetupImpl const* = nullptr) const;

      template <typename DataT>
      void getImplementation(DataT const*& iData,
                             char const* iName,
                             ComponentDescription const*& iDesc,
                             bool iTransientAccessOnly,
                             std::shared_ptr<ESHandleExceptionFactory>& whyFailedFactory,
                             EventSetupImpl const* iEventSetupImpl) const {
        DataKey dataKey(DataKey::makeTypeTag<DataT>(), iName, DataKey::kDoNotCopyMemory);

        void const* pValue = this->getFromProxy(dataKey, iDesc, iTransientAccessOnly, iEventSetupImpl);
        if (nullptr == pValue) {
          whyFailedFactory = makeESHandleExceptionFactory([=] {
            NoProxyException<DataT> ex(this->key(), dataKey);
            return std::make_exception_ptr(ex);
          });
        }
        iData = reinterpret_cast<DataT const*>(pValue);
      }

      template <typename DataT>
      void getImplementation(DataT const*& iData,
                             ESProxyIndex iProxyIndex,
                             bool iTransientAccessOnly,
                             ComponentDescription const*& oDesc,
                             std::shared_ptr<ESHandleExceptionFactory>& whyFailedFactory,
                             EventSetupImpl const* iEventSetupImpl) const {
        DataKey const* dataKey = nullptr;
        if (iProxyIndex.value() == std::numeric_limits<int>::max()) {
          whyFailedFactory = makeESHandleExceptionFactory([=] {
            NoProxyException<DataT> ex(this->key(), {});
            return std::make_exception_ptr(ex);
          });
          iData = nullptr;
          return;
        }
        assert(iProxyIndex.value() > -1 and
               iProxyIndex.value() < static_cast<ESProxyIndex::Value_t>(keysForProxies_.size()));
        void const* pValue = this->getFromProxy(iProxyIndex, iTransientAccessOnly, oDesc, dataKey, iEventSetupImpl);
        if (nullptr == pValue) {
          whyFailedFactory = makeESHandleExceptionFactory([=] {
            NoProxyException<DataT> ex(this->key(), *dataKey);
            return std::make_exception_ptr(ex);
          });
        }
        iData = reinterpret_cast<DataT const*>(pValue);
      }

      // ---------- member data --------------------------------

      // We allow the validity to be modified while it is being used in
      // the case where a new sync value is being initialized and the
      // first part of the new validity range is the same,
      // but the last part of the validity range changes. In this case,
      // we do not want to start a new IOV, all the cached data should
      // remain valid. The atomic bool validityModificationUnderway_
      // protects access to validity_ while this change is made.
      CMS_THREAD_SAFE mutable ValidityInterval validity_;

      EventSetupRecordKey key_;
      std::vector<DataKey> keysForProxies_;
      std::vector<edm::propagate_const<DataProxy*>> proxies_;
      ActivityRegistry const* activityRegistry_;
      unsigned long long cacheIdentifier_;
      unsigned int iovIndex_;
      std::atomic<bool> isAvailable_;
      mutable std::atomic<bool> validityModificationUnderway_;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
