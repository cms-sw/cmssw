#ifndef FWCore_Framework_EventSetupRecord_h
#define FWCore_Framework_EventSetupRecord_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecord
//
/**\class edm::eventsetup::EventSetupRecord

 Description: Base class for all Records in an EventSetup.  A record is associated
 with data with the same lifetime.

 Usage:

 This class contains the interface that clients of the EventSetup
 use to get data associated with a record.

 This class holds a pointer to an EventSetupRecordImpl class, which
 is the class used to get the DataProxies associated with a given Record.
 It also has a pointer to the EventSetupImpl which is used to lookup
 dependent records in DependentRecordImplementation.

 It is important for the management of the Records that each Record
 know the ValidityInterval that represents the time over which its
 data is valid. The ValidityInterval is set by its EventSetupRecordProvider
 using the 'set' function.  This quantity can be recovered
 through the 'validityInterval' method.

 Most of the time one uses a record obtained from an EventSetup object
 and in that case the pointers this contains will be properly initialized
 automatically. If you construct a record object directly (usually using
 a type derived from this), then only a very limited subset of functions
 can be called because most of the member functions will dereference one
 of the null pointers and seg fault.
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
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Utilities/interface/Likely.h"

// system include files
#include <exception>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <atomic>
#include <cassert>
#include <limits>

// forward declarations
namespace cms {
  class Exception;
}

class testEventsetup;
class testEventsetupRecord;

namespace edm {
  template <typename T>
  class ESHandle;
  class ESHandleExceptionFactory;
  class ESInputTag;
  class EventSetupImpl;

  namespace eventsetup {
    struct ComponentDescription;
    class DataProxy;
    class EventSetupRecordKey;

    class EventSetupRecord {
    public:
      EventSetupRecord();
      EventSetupRecord(EventSetupRecord&&) = default;
      EventSetupRecord& operator=(EventSetupRecord&&) = default;

      EventSetupRecord(EventSetupRecord const&) = default;
      EventSetupRecord& operator=(EventSetupRecord const&) = default;
      virtual ~EventSetupRecord();

      // ---------- const member functions ---------------------
      ValidityInterval validityInterval() const { return impl_->validityInterval(); }

      void setImpl(EventSetupRecordImpl const* iImpl,
                   unsigned int transitionID,
                   ESProxyIndex const* getTokenIndices,
                   EventSetupImpl const* iEventSetupImpl,
                   bool requireTokens) {
        impl_ = iImpl;
        transitionID_ = transitionID;
        getTokenIndices_ = getTokenIndices;
        eventSetupImpl_ = iEventSetupImpl;
        requireTokens_ = requireTokens;
      }

      template <typename HolderT>
      bool get(HolderT& iHolder) const {
        return get("", iHolder);
      }

      template <typename HolderT>
      bool get(char const* iName, HolderT& iHolder) const {
        if
          UNLIKELY(requireTokens_) {
            throwCalledGetWithoutToken(heterocontainer::className<typename HolderT::value_type>(), iName);
          }
        typename HolderT::value_type const* value = nullptr;
        ComponentDescription const* desc = nullptr;
        std::shared_ptr<ESHandleExceptionFactory> whyFailedFactory;
        impl_->getImplementation(value, iName, desc, iHolder.transientAccessOnly, whyFailedFactory, eventSetupImpl_);

        if (value) {
          iHolder = HolderT(value, desc);
          return true;
        } else {
          iHolder = HolderT(std::move(whyFailedFactory));
          return false;
        }
      }

      template <typename HolderT>
      bool get(std::string const& iName, HolderT& iHolder) const {
        return get(iName.c_str(), iHolder);
      }

      template <typename HolderT>
      bool get(ESInputTag const& iTag, HolderT& iHolder) const {
        if
          UNLIKELY(requireTokens_) {
            throwCalledGetWithoutToken(heterocontainer::className<typename HolderT::value_type>(), iTag.data().c_str());
          }
        typename HolderT::value_type const* value = nullptr;
        ComponentDescription const* desc = nullptr;
        std::shared_ptr<ESHandleExceptionFactory> whyFailedFactory;
        impl_->getImplementation(
            value, iTag.data().c_str(), desc, iHolder.transientAccessOnly, whyFailedFactory, eventSetupImpl_);

        if (value) {
          validate(desc, iTag);
          iHolder = HolderT(value, desc);
          return true;
        } else {
          iHolder = HolderT(std::move(whyFailedFactory));
          return false;
        }
      }

      ///returns false if no data available for key
      bool doGet(DataKey const& aKey, bool aGetTransiently = false) const;

      /**returns true only if someone has already requested data for this key
          and the data was retrieved
          */
      bool wasGotten(DataKey const& aKey) const;

      /**returns the ComponentDescription for the module which creates the data or 0
          if no module has been registered for the data. This does not cause the data to
          actually be constructed.
          */
      ComponentDescription const* providerDescription(DataKey const& aKey) const;

      virtual EventSetupRecordKey key() const = 0;

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
      unsigned long long cacheIdentifier() const { return impl_->cacheIdentifier(); }

      /**When you are processing multiple validity intervals concurrently,
          each must have its own cache for data. These caches are numbered
          from 0 to one less than the maximum number of concurrent validity
          intervals allowed for that record. This number is returned by
          the following function. For one record type, all the different validity
          intervals being processed at the same time will have a different
          iovIndex. But the iovIndex's of record types that are different
          are independent of each other.
          */
      unsigned int iovIndex() const { return impl_->iovIndex(); }

      ///clears the oToFill vector and then fills it with the keys for all registered data keys
      void fillRegisteredDataKeys(std::vector<DataKey>& oToFill) const { impl_->fillRegisteredDataKeys(oToFill); }

      ///Classes that derive from EventSetupRecord can redefine this with a false value
      static constexpr bool allowConcurrentIOVs_ = true;

      friend class ::testEventsetup;
      friend class ::testEventsetupRecord;

    protected:
      template <template <typename> typename H, typename T, typename R>
      H<T> getHandleImpl(ESGetToken<T, R> const& iToken) const {
        if
          UNLIKELY(iToken.transitionID() != transitionID()) { throwWrongTransitionID(); }
        assert(iToken.isInitialized());
        assert(getTokenIndices_);
        //need to check token has valid index
        if
          UNLIKELY(not iToken.hasValidIndex()) { return invalidTokenHandle<H>(iToken); }

        auto proxyIndex = getTokenIndices_[iToken.index().value()];
        if
          UNLIKELY(proxyIndex.value() == std::numeric_limits<int>::max()) { return noProxyHandle<H>(iToken); }

        T const* value = nullptr;
        ComponentDescription const* desc = nullptr;
        std::shared_ptr<ESHandleExceptionFactory> whyFailedFactory;

        impl_->getImplementation(value, proxyIndex, H<T>::transientAccessOnly, desc, whyFailedFactory, eventSetupImpl_);

        if
          UNLIKELY(not value) { return H<T>(std::move(whyFailedFactory)); }
        return H<T>(value, desc);
      }

      EventSetupImpl const& eventSetup() const { return *eventSetupImpl_; }

      ESProxyIndex const* getTokenIndices() const { return getTokenIndices_; }

      void validate(ComponentDescription const*, ESInputTag const&) const;

      void addTraceInfoToCmsException(cms::Exception& iException,
                                      char const* iName,
                                      ComponentDescription const*,
                                      DataKey const&) const;

      EventSetupRecordImpl const* impl() const { return impl_; }

      unsigned int transitionID() const { return transitionID_; }

      bool requireTokens() const { return requireTokens_; }

    private:
      template <template <typename> typename H, typename T, typename R>
      H<T> invalidTokenHandle(ESGetToken<T, R> const& iToken) const {
        auto const key = this->key();
        return H<T>{
            makeESHandleExceptionFactory([key] { return makeInvalidTokenException(key, DataKey::makeTypeTag<T>()); })};
      }

      template <template <typename> typename H, typename T, typename R>
      H<T> noProxyHandle(ESGetToken<T, R> const& iToken) const {
        auto const key = this->key();
        auto name = iToken.name();
        return H<T>{makeESHandleExceptionFactory([key, name] {
          NoProxyException<T> ex(key, DataKey{DataKey::makeTypeTag<T>(), name});
          return std::make_exception_ptr(ex);
        })};
      }

      void const* getFromProxy(DataKey const& iKey,
                               ComponentDescription const*& iDesc,
                               bool iTransientAccessOnly) const;

      static std::exception_ptr makeInvalidTokenException(EventSetupRecordKey const&, TypeTag const&);
      void throwWrongTransitionID() const;
      static void throwCalledGetWithoutToken(const char* iTypeName, const char* iLabel);
      // ---------- member data --------------------------------
      EventSetupRecordImpl const* impl_ = nullptr;
      EventSetupImpl const* eventSetupImpl_ = nullptr;
      ESProxyIndex const* getTokenIndices_ = nullptr;
      unsigned int transitionID_ = std::numeric_limits<unsigned int>::max();
      bool requireTokens_ = false;
    };

    class EventSetupRecordGeneric : public EventSetupRecord {
    public:
      EventSetupRecordGeneric(EventSetupRecordImpl const* iImpl,
                              unsigned int iTransitionID,
                              ESProxyIndex const* getTokenIndices,
                              EventSetupImpl const* eventSetupImpl,
                              bool requireTokens = false) {
        setImpl(iImpl, iTransitionID, getTokenIndices, eventSetupImpl, requireTokens);
      }

      EventSetupRecordKey key() const final { return impl()->key(); }
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
