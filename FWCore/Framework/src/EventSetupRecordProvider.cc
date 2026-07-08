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
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/SupportingRecordIntervalFinderHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "make_shared_noexcept_false.h"

namespace edm {
  namespace eventsetup {

    EventSetupRecordProvider::EventSetupRecordProvider(const EventSetupRecordKey& iKey,
                                                       ActivityRegistry const* activityRegistry,
                                                       unsigned int nConcurrentIOVs)
        : key_(iKey),
          validityInterval_(),
          finder_(),
          providers_(),
          multipleFinders_(new std::vector<edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>>>()),
          nConcurrentIOVs_(nConcurrentIOVs) {
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

    void EventSetupRecordProvider::addFinder(std::shared_ptr<EventSetupRecordIntervalFinder> iFinder) {
      auto oldFinder = finder();
      finder_ = iFinder;
      if (nullptr != multipleFinders_.get()) {
        multipleFinders_->emplace_back(iFinder);
      } else {
        //dependent records set there finders after the multipleFinders_ has been released
        // but they also have never had a finder set
        if (nullptr != oldFinder.get()) {
          cms::Exception("EventSetupMultipleSources")
              << "An additional source has been added to the Record " << key_.name() << "'\n"
              << "after all the other sources have been dealt with.  This is a logic error, please send email to the "
                 "framework group.";
        }
      }
    }

    void EventSetupRecordProvider::setSupportingProviders(
        const std::map<EventSetupRecordKey, std::shared_ptr<EventSetupRecordProvider>>& iKeyToProviders) {
      auto const& supportingKeys = supportingRecords();
      if (supportingKeys.empty()) {
        return;
      }
      assert(not(finder_ and dynamic_cast<DependentRecordIntervalFinder*>(finder_.get()) != nullptr));
      std::shared_ptr<DependentRecordIntervalFinder> newFinder =
          make_shared_noexcept_false<DependentRecordIntervalFinder>(key());

      std::shared_ptr<EventSetupRecordIntervalFinder> old = swapFinder(newFinder);

      std::string missingRecords;

      for (auto const& key : supportingKeys) {
        auto itFound = iKeyToProviders.find(key);
        if (itFound == iKeyToProviders.end()) {
          if (missingRecords.empty()) {
            missingRecords = key.name();
          } else {
            missingRecords += ", ";
            missingRecords += key.name();
          }
        } else {
          SupportingRecordIntervalFinderHelper helper(itFound->first, itFound->second->finder());
          newFinder->addSupporter(helper);
        }
      }
      if (!missingRecords.empty()) {
        edm::LogInfo("EventSetupDependency")
            << "The EventSetup record " << key().name() << " depends on at least one Record \n (" << missingRecords
            << ") which is not present in the job."
               "\n This may lead to an exception begin thrown during event processing.\n If no exception occurs "
               "during the job than it is usually safe to ignore this message.";
      }

      //if a finder was already set, add it as a dependency.  This is done to ensure that the IOVs properly change even if the
      // old finder does not update each time a supporting record does change
      if (old.get() != nullptr) {
        newFinder->setAlternateFinder(old);
      }
    }
    void EventSetupRecordProvider::usePreferred(const DataToPreferredProviderMap& iMap) {
      using std::placeholders::_1;
      for_all(providers_, std::bind(&EventSetupRecordProvider::addResolversToRecordHelper, this, _1, iMap));
      if (1 < multipleFinders_->size()) {
        std::shared_ptr<IntersectingIOVRecordIntervalFinder> intFinder =
            make_shared_noexcept_false<IntersectingIOVRecordIntervalFinder>(key_);
        intFinder->swapFinders(*multipleFinders_);
        finder_ = intFinder;
      }
      //now we get rid of the temporary
      multipleFinders_.reset(nullptr);
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

    void EventSetupRecordProvider::initializeForNewIOV(unsigned int iovIndex,
                                                       unsigned long long cacheIdentifier,
                                                       EventSetupImpl& eventSetupImpl) {
      EventSetupRecordImpl* impl = &recordImpls_[iovIndex];
      recordImpl_ = impl;
      impl->initializeForNewIOV(cacheIdentifier, validityInterval_);
      eventSetupImpl.addRecordImpl(*recordImpl_);
    }

    void EventSetupRecordProvider::continueIOV(bool newEventSetupImpl, EventSetupImpl& eventSetupImpl) {
      if (intervalStatus_ == IntervalStatus::UpdateIntervalEnd) {
        recordImpl_->setSafely(validityInterval_);
      }
      if (newEventSetupImpl && intervalStatus_ != IntervalStatus::Invalid) {
        eventSetupImpl.addRecordImpl(*recordImpl_);
      }
    }

    void EventSetupRecordProvider::endIOV(unsigned int iovIndex) { recordImpls_[iovIndex].invalidateResolvers(); }

    bool EventSetupRecordProvider::setValidityIntervalFor(const IOVSyncValue& iTime) {
      intervalStatus_ = IntervalStatus::Invalid;

      if (validityInterval_.first() != IOVSyncValue::invalidIOVSyncValue() && validityInterval_.validFor(iTime)) {
        intervalStatus_ = IntervalStatus::SameInterval;

      } else if (finder_.get() != nullptr) {
        IOVSyncValue oldFirst(validityInterval_.first());
        IOVSyncValue oldLast(validityInterval_.last());
        validityInterval_ = finder_->findIntervalFor(key_, iTime);

        // An interval is valid if and only if the start of the interval is
        // valid. If the start is valid and the end is invalid, it means we
        // do not know when the interval ends, but the interval is valid and
        // iTime is within the interval.
        if (validityInterval_.first() != IOVSyncValue::invalidIOVSyncValue()) {
          // An interval is new if the start of the interval changes
          if (validityInterval_.first() != oldFirst) {
            intervalStatus_ = IntervalStatus::NewInterval;

            // If the start is the same but the end changes, we consider
            // this the same interval because we do not want to do the
            // work to update the caches of data in this case.
          } else if (validityInterval_.last() != oldLast) {
            intervalStatus_ = IntervalStatus::UpdateIntervalEnd;
          } else {
            intervalStatus_ = IntervalStatus::SameInterval;
          }
        }
      }
      return intervalStatus_ != IntervalStatus::Invalid;
    }

    void EventSetupRecordProvider::resetResolvers() {
      // Clear out all the ESProductResolver's
      for (auto& recordImplIter : recordImpls_) {
        recordImplIter.invalidateResolvers();
        recordImplIter.resetIfTransientInResolvers();
      }
      // Force a new IOV to start with a new cacheIdentifier
      // on the next eventSetupForInstance call.
      validityInterval_ = ValidityInterval{};
      if (finder_.get() != nullptr) {
        finder_->resetInterval(key_);
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
