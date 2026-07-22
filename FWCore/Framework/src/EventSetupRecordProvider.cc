// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordProvider
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Fri Mar 25 19:12:03 EST 2005
//

// system include files
#include <algorithm>
#include <cassert>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"

#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "make_shared_noexcept_false.h"

namespace edm {
  namespace eventsetup {

    EventSetupRecordProvider::EventSetupRecordProvider(const EventSetupRecordKey& iKey,
                                                       ActivityRegistry const* activityRegistry,
                                                       unsigned int nConcurrentIOVs)
        : key_(iKey), providers_(), nConcurrentIOVs_(nConcurrentIOVs) {
      recordImpls_.reserve(nConcurrentIOVs);
      for (unsigned int i = 0; i < nConcurrentIOVs_; ++i) {
        recordImpls_.emplace_back(iKey, activityRegistry, i);
      }
    }

    EventSetupRecordImpl const& EventSetupRecordProvider::firstRecordImpl() const { return recordImpls_[0]; }

    void EventSetupRecordProvider::add(std::shared_ptr<ESProductResolverProvider> iProvider) {
      assert(iProvider->isUsingRecord(key_));
      edm::propagate_const<std::shared_ptr<ESProductResolverProvider>> pProvider(iProvider);
      assert(!search_all(providers_, pProvider));
      providers_.emplace_back(iProvider);
    }

    void EventSetupRecordProvider::usePreferred(const DataToPreferredProviderMap& iMap) {
      using std::placeholders::_1;
      for_all(providers_, std::bind(&EventSetupRecordProvider::addResolversToRecordHelper, this, _1, iMap));
    }

    void EventSetupRecordProvider::addResolversToRecord(
        std::shared_ptr<ESProductResolverProvider> iProvider,
        const EventSetupRecordProvider::DataToPreferredProviderMap& iMap) {
      typedef ESProductResolverProvider::KeyedResolvers ResolverList;
      typedef EventSetupRecordProvider::DataToPreferredProviderMap PreferredMap;

      for (auto& record : recordImpls_) {
        ResolverList& keyedResolvers(iProvider->keyedResolvers(this->key(), record.iovIndex()));

        for (auto keyedResolver : keyedResolvers) {
          PreferredMap::const_iterator itFound = iMap.find(keyedResolver.dataKey_);
          if (iMap.end() != itFound) {
            if (itFound->second.type_ != keyedResolver.productResolver_->providerDescription()->type_ ||
                itFound->second.label_ != keyedResolver.productResolver_->providerDescription()->label_) {
              //this is not the preferred provider
              continue;
            }
          }
          record.add(keyedResolver.dataKey_, keyedResolver.productResolver_);
        }
      }
    }

    EventSetupRecordImpl const& EventSetupRecordProvider::initializeForNewIOV(
        unsigned int iovIndex, unsigned long long cacheIdentifier, ValidityInterval const& validityInterval) {
      EventSetupRecordImpl& impl = recordImpls_[iovIndex];
      impl.initializeForNewIOV(cacheIdentifier, validityInterval);
      return impl;
    }

    EventSetupRecordImpl const& EventSetupRecordProvider::continueIOV(unsigned int iovIndex,
                                                                      edm::ValidityInterval const* validityInterval) {
      assert(iovIndex < recordImpls_.size());
      auto const& recordImpl = recordImpls_[iovIndex];

      if (validityInterval) {
        recordImpl.setSafely(*validityInterval);
      }
      return recordImpl;
    }

    void EventSetupRecordProvider::endIOV(unsigned int iovIndex) { recordImpls_[iovIndex].invalidateResolvers(); }

    void EventSetupRecordProvider::resetResolvers() {
      // Clear out all the ESProductResolver's
      for (auto& recordImplIter : recordImpls_) {
        recordImplIter.invalidateResolvers();
        recordImplIter.resetIfTransientInResolvers();
      }
    }

    void EventSetupRecordProvider::getReferencedESProducers(
        std::map<EventSetupRecordKey, std::vector<ComponentDescription const*>>& referencedESProducers) const {
      firstRecordImpl().getESProducers(referencedESProducers[key_]);
    }

    void EventSetupRecordProvider::fillAllESProductResolverProviders(
        std::vector<ESProductResolverProvider const*>& allESProductResolverProviders,
        std::unordered_set<unsigned int>& componentIDs) const {
      for (auto const& provider : providers_) {
        if (componentIDs.insert(provider->description().id_).second) {
          allESProductResolverProviders.push_back(provider.get());
        }
      }
    }

    void EventSetupRecordProvider::updateLookup(ESRecordsToProductResolverIndices const& iResolverToIndices) {
      for (auto& productResolverProvider : providers_) {
        productResolverProvider->updateLookup(iResolverToIndices);
      }
    }

    std::set<EventSetupRecordKey> EventSetupRecordProvider::supportingRecords() const { return dependencies(key()); }

    std::set<ComponentDescription> EventSetupRecordProvider::resolverProviderDescriptions() const {
      using std::placeholders::_1;
      std::set<ComponentDescription> descriptions;
      std::transform(providers_.begin(),
                     providers_.end(),
                     std::inserter(descriptions, descriptions.end()),
                     std::bind(&ESProductResolverProvider::description, _1));
      return descriptions;
    }

    std::shared_ptr<ESProductResolverProvider> EventSetupRecordProvider::resolverProvider(
        ComponentDescription const& iDesc) {
      using std::placeholders::_1;
      auto itFound = std::find_if(
          providers_.begin(),
          providers_.end(),
          std::bind(
              std::equal_to<ComponentDescription>(), iDesc, std::bind(&ESProductResolverProvider::description, _1)));
      if (itFound == providers_.end()) {
        return std::shared_ptr<ESProductResolverProvider>();
      }
      return get_underlying_safe(*itFound);
    }

    std::vector<DataKey> const& EventSetupRecordProvider::registeredDataKeys() const {
      return firstRecordImpl().registeredDataKeys();
    }

    std::vector<ComponentDescription const*> EventSetupRecordProvider::componentsForRegisteredDataKeys() const {
      return firstRecordImpl().componentsForRegisteredDataKeys();
    }

    std::vector<unsigned int> EventSetupRecordProvider::produceMethodIDsForRegisteredDataKeys() const {
      return firstRecordImpl().produceMethodIDsForRegisteredDataKeys();
    }

  }  // namespace eventsetup
}  // namespace edm
