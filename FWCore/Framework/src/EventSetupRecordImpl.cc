// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordImpl
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sat Mar 26 18:06:32 EST 2005
//

// system include files
#include <algorithm>
#include <cassert>
#include <iterator>
#include <sstream>
#include <utility>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  namespace eventsetup {

    EventSetupRecordImpl::EventSetupRecordImpl(EventSetupRecordKey const& iKey,
                                               ActivityRegistry const* activityRegistry,
                                               unsigned int iovIndex)
        : validity_(),
          key_(iKey),
          activityRegistry_(activityRegistry),
          cacheIdentifier_(1),  // initial value meaningless, gets overwritten before use
          iovIndex_(iovIndex),
          isAvailable_(true),
          validityModificationUnderway_(false) {}

    EventSetupRecordImpl::EventSetupRecordImpl(EventSetupRecordImpl&& source)
        : validity_{source.validity_},
          key_{source.key_},
          keysForProxies_{std::move(source.keysForProxies_)},
          proxies_(std::move(source.proxies_)),
          activityRegistry_{source.activityRegistry_},
          cacheIdentifier_{source.cacheIdentifier_},
          iovIndex_{source.iovIndex_},
          isAvailable_{source.isAvailable_.load()},
          validityModificationUnderway_{source.validityModificationUnderway_.load()} {}

    EventSetupRecordImpl& EventSetupRecordImpl::operator=(EventSetupRecordImpl&& rhs) {
      validity_ = rhs.validity_;
      key_ = rhs.key_;
      keysForProxies_ = std::move(rhs.keysForProxies_);
      proxies_ = std::move(rhs.proxies_);
      activityRegistry_ = rhs.activityRegistry_;
      cacheIdentifier_ = rhs.cacheIdentifier_;
      iovIndex_ = rhs.iovIndex_;
      isAvailable_.store(rhs.isAvailable_.load());
      validityModificationUnderway_.store(validityModificationUnderway_.load());
      return *this;
    }

    ValidityInterval EventSetupRecordImpl::validityInterval() const {
      bool expected = false;
      while (not validityModificationUnderway_.compare_exchange_strong(expected, true)) {
        expected = false;
      }
      ValidityInterval temp = validity_;
      validityModificationUnderway_ = false;
      return temp;
    }

    void EventSetupRecordImpl::initializeForNewIOV(unsigned long long iCacheIdentifier,
                                                   ValidityInterval const& iValidityInterval,
                                                   bool hasFinder) {
      cacheIdentifier_ = iCacheIdentifier;
      validity_ = iValidityInterval;
      if (hasFinder) {
        for (auto& dataProxy : proxies_) {
          dataProxy->initializeForNewIOV();
        }
      }
    }

    void EventSetupRecordImpl::setSafely(const ValidityInterval& iInterval) const {
      bool expected = false;
      while (not validityModificationUnderway_.compare_exchange_strong(expected, true)) {
        expected = false;
      }
      validity_ = iInterval;
      validityModificationUnderway_ = false;
    }

    void EventSetupRecordImpl::getESProducers(std::vector<ComponentDescription const*>& esproducers) const {
      esproducers.clear();
      esproducers.reserve(proxies_.size());
      for (auto const& iData : proxies_) {
        ComponentDescription const* componentDescription = iData->providerDescription();
        if (!componentDescription->isLooper_ && !componentDescription->isSource_) {
          esproducers.push_back(componentDescription);
        }
      }
    }

    std::vector<ComponentDescription const*> EventSetupRecordImpl::componentsForRegisteredDataKeys() const {
      std::vector<ComponentDescription const*> ret;
      ret.reserve(proxies_.size());
      for (auto const& proxy : proxies_) {
        ret.push_back(proxy->providerDescription());
      }
      return ret;
    }

    bool EventSetupRecordImpl::add(const DataKey& iKey, DataProxy* iProxy) {
      const DataProxy* proxy = find(iKey);
      if (nullptr != proxy) {
        //
        // we already know the field exist, so do not need to check against end()
        //

        // POLICY: If a Producer and a Source both claim to deliver the same data, the
        //  Producer 'trumps' the Source. If two modules of the same type claim to deliver the
        //  same data, this is an error unless the configuration specifically states which one
        //  is to be chosen.  A Looper trumps both a Producer and a Source.

        assert(proxy->providerDescription());
        assert(iProxy->providerDescription());
        if (iProxy->providerDescription()->isLooper_) {
          proxies_[std::distance(keysForProxies_.begin(),
                                 std::lower_bound(keysForProxies_.begin(), keysForProxies_.end(), iKey))] = iProxy;
          return true;
        }

        if (proxy->providerDescription()->isSource_ == iProxy->providerDescription()->isSource_) {
          //should lookup to see if there is a specified 'chosen' one and only if not, throw the exception
          throw cms::Exception("EventSetupConflict")
              << "two EventSetup " << (proxy->providerDescription()->isSource_ ? "Sources" : "Producers")
              << " want to deliver type=\"" << iKey.type().name() << "\" label=\"" << iKey.name().value() << "\"\n"
              << " from record " << key().type().name() << ". The two providers are \n"
              << "1) type=\"" << proxy->providerDescription()->type_ << "\" label=\""
              << proxy->providerDescription()->label_ << "\"\n"
              << "2) type=\"" << iProxy->providerDescription()->type_ << "\" label=\""
              << iProxy->providerDescription()->label_ << "\"\n"
              << "Please either\n   remove one of these "
              << (proxy->providerDescription()->isSource_ ? "Sources" : "Producers")
              << "\n   or find a way of configuring one of them so it does not deliver this data"
              << "\n   or use an es_prefer statement in the configuration to choose one.";
        } else if (proxy->providerDescription()->isSource_) {
          proxies_[std::distance(keysForProxies_.begin(),
                                 std::lower_bound(keysForProxies_.begin(), keysForProxies_.end(), iKey))] = iProxy;
        } else {
          return false;
        }
      } else {
        auto lb = std::lower_bound(keysForProxies_.begin(), keysForProxies_.end(), iKey);
        auto index = std::distance(keysForProxies_.begin(), lb);
        keysForProxies_.insert(lb, iKey);
        proxies_.insert(proxies_.begin() + index, iProxy);
      }
      return true;
    }

    void EventSetupRecordImpl::clearProxies() {
      keysForProxies_.clear();
      proxies_.clear();
    }

    void EventSetupRecordImpl::invalidateProxies() {
      for (auto& dataProxy : proxies_) {
        dataProxy->invalidate();
      }
    }

    void EventSetupRecordImpl::resetIfTransientInProxies() {
      for (auto& dataProxy : proxies_) {
        dataProxy->resetIfTransient();
      }
    }

    void const* EventSetupRecordImpl::getFromProxyAfterPrefetch(ESProxyIndex iProxyIndex,
                                                                bool iTransientAccessOnly,
                                                                ComponentDescription const*& iDesc,
                                                                DataKey const*& oGottenKey) const {
      const DataProxy* proxy = proxies_[iProxyIndex.value()];
      assert(nullptr != proxy);
      iDesc = proxy->providerDescription();

      auto const& key = keysForProxies_[iProxyIndex.value()];
      oGottenKey = &key;

      void const* hold = nullptr;
      try {
        convertException::wrap([&]() { hold = proxy->getAfterPrefetch(*this, key, iTransientAccessOnly); });
      } catch (cms::Exception& e) {
        addTraceInfoToCmsException(e, key.name().value(), proxy->providerDescription(), key);
        throw;
      }
      return hold;
    }

    const DataProxy* EventSetupRecordImpl::find(const DataKey& iKey) const {
      auto lb = std::lower_bound(keysForProxies_.begin(), keysForProxies_.end(), iKey);
      if ((lb == keysForProxies_.end()) or (*lb != iKey)) {
        return nullptr;
      }
      return proxies_[std::distance(keysForProxies_.begin(), lb)].get();
    }

    void EventSetupRecordImpl::prefetchAsync(WaitingTaskHolder iTask,
                                             ESProxyIndex iProxyIndex,
                                             EventSetupImpl const* iEventSetupImpl,
                                             ServiceToken const& iToken,
                                             ESParentContext iParent) const {
      if UNLIKELY (iProxyIndex.value() == std::numeric_limits<int>::max()) {
        return;
      }

      const DataProxy* proxy = proxies_[iProxyIndex.value()];
      if (nullptr != proxy) {
        auto const& key = keysForProxies_[iProxyIndex.value()];
        proxy->prefetchAsync(iTask, *this, key, iEventSetupImpl, iToken, iParent);
      }
    }

    bool EventSetupRecordImpl::wasGotten(const DataKey& aKey) const {
      const DataProxy* proxy = find(aKey);
      if (nullptr != proxy) {
        return proxy->cacheIsValid();
      }
      return false;
    }

    edm::eventsetup::ComponentDescription const* EventSetupRecordImpl::providerDescription(const DataKey& aKey) const {
      const DataProxy* proxy = find(aKey);
      if (nullptr != proxy) {
        return proxy->providerDescription();
      }
      return nullptr;
    }

    void EventSetupRecordImpl::fillRegisteredDataKeys(std::vector<DataKey>& oToFill) const {
      oToFill = keysForProxies_;
    }

    void EventSetupRecordImpl::addTraceInfoToCmsException(cms::Exception& iException,
                                                          const char* iName,
                                                          const ComponentDescription* iDescription,
                                                          const DataKey& iKey) const {
      std::ostringstream ost;
      ost << "Using EventSetup component " << iDescription->type_ << "/'" << iDescription->label_ << "' to make data "
          << iKey.type().name() << "/'" << iName << "' in record " << this->key().type().name();
      iException.addContext(ost.str());
    }

  }  // namespace eventsetup
}  // namespace edm
