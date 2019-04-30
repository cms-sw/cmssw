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

#include "FWCore/Framework/interface/ParameterSetIDHolder.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "make_shared_noexcept_false.h"

namespace edm {
   namespace eventsetup {

EventSetupRecordProvider::EventSetupRecordProvider(const EventSetupRecordKey& iKey,
                                                   ActivityRegistry const* activityRegistry,
                                                   unsigned int nConcurrentIOVs) :
    key_(iKey),
    nConcurrentIOVs_(nConcurrentIOVs),
    validityInterval_(),
    finder_(),
    providers_(),
    multipleFinders_(new std::vector<edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>>>())
{
  for (unsigned int i = 0; i < nConcurrentIOVs_; ++i) {
    recordImpls_.push_back(std::make_unique<EventSetupRecordImpl>(iKey, activityRegistry, i));
  }
}

EventSetupRecordImpl const& EventSetupRecordProvider::firstRecordImpl() const {
  return *recordImpls_[0];
}

void
EventSetupRecordProvider::add(std::shared_ptr<DataProxyProvider> iProvider)
{
   assert(iProvider->isUsingRecord(key_));
   edm::propagate_const<std::shared_ptr<DataProxyProvider>> pProvider(iProvider);
   assert(!search_all(providers_, pProvider));
   providers_.emplace_back(iProvider);
}

void 
EventSetupRecordProvider::addFinder(std::shared_ptr<EventSetupRecordIntervalFinder> iFinder)
{
   auto oldFinder = finder();
   finder_ = iFinder;
   if (nullptr != multipleFinders_.get()) {
     multipleFinders_->emplace_back(iFinder);
   } else {
     //dependent records set there finders after the multipleFinders_ has been released
     // but they also have never had a finder set
     if(nullptr != oldFinder.get()) {
       cms::Exception("EventSetupMultipleSources")<<"An additional source has been added to the Record "
       <<key_.name()<<"'\n"
       <<"after all the other sources have been dealt with.  This is a logic error, please send email to the framework group.";
     }
   }
}
void
EventSetupRecordProvider::setValidityInterval(const ValidityInterval& iInterval)
{
   validityInterval_ = iInterval;
   initializeForNewSyncValue();
}

void 
EventSetupRecordProvider::setDependentProviders(const std::vector< std::shared_ptr<EventSetupRecordProvider> >& iProviders)
{
   using std::placeholders::_1;
   std::shared_ptr<DependentRecordIntervalFinder> newFinder = make_shared_noexcept_false<DependentRecordIntervalFinder>(key());

   std::shared_ptr<EventSetupRecordIntervalFinder> old = swapFinder(newFinder);

   for(auto const& p: iProviders) { newFinder->addProviderWeAreDependentOn(p); };
   //if a finder was already set, add it as a depedency.  This is done to ensure that the IOVs properly change even if the
   // old finder does not update each time a dependent record does change
   if(old.get() != nullptr) {
      newFinder->setAlternateFinder(old);
   }
}
void 
EventSetupRecordProvider::usePreferred(const DataToPreferredProviderMap& iMap)
{
  using std::placeholders::_1;
  for_all(providers_, std::bind(&EventSetupRecordProvider::addProxiesToRecordHelper,this,_1,iMap));
  if (1 < multipleFinders_->size()) {
     std::shared_ptr<IntersectingIOVRecordIntervalFinder> intFinder = make_shared_noexcept_false<IntersectingIOVRecordIntervalFinder>(key_);
     intFinder->swapFinders(*multipleFinders_);
     finder_ = intFinder;
     hasLegacyESSource_ = intFinder->hasLegacyESSource();
  } else {
     if (finder_.get() != nullptr) {
        hasLegacyESSource_ = finder_->legacyESSource();
     }
  }

  //now we get rid of the temporary
  multipleFinders_.reset(nullptr);
}

void 
EventSetupRecordProvider::addProxiesToRecord(std::shared_ptr<DataProxyProvider> iProvider,
                                const EventSetupRecordProvider::DataToPreferredProviderMap& iMap) {
   typedef DataProxyProvider::KeyedProxies ProxyList ;
   typedef EventSetupRecordProvider::DataToPreferredProviderMap PreferredMap;
   
   for (unsigned int iovIndex = 0; iovIndex < nConcurrentIOVs_; ++iovIndex) {

      ProxyList& keyedProxies(iProvider->keyedProxies(this->key(), iovIndex));

      for (auto& keyedProxy : keyedProxies) {

         PreferredMap::const_iterator itFound = iMap.find(keyedProxy.first);
         if(iMap.end() != itFound) {
            if ( itFound->second.type_ != keyedProxy.second->providerDescription()->type_ ||
                 itFound->second.label_ != keyedProxy.second->providerDescription()->label_ ) {
               //this is not the preferred provider
               continue;
            }
         }
         recordImpls_.at(iovIndex)->add(keyedProxy.first, keyedProxy.second.get()) ;
      }
   }
}

void
EventSetupRecordProvider::initializeForNewIOV(unsigned int iovIndex,
                                              unsigned long long cacheIdentifier) {
  EventSetupRecordImpl* impl = recordImpls_[iovIndex].get();
  recordImpl_ = impl;
  impl->set(validityInterval_);
  impl->setCacheIdentifier(cacheIdentifier);
  eventSetupImpl_->addRecordImpl(*recordImpl_);
}

void
EventSetupRecordProvider::continueIOV(bool newEventSetupImpl) {
  if (intervalStatus_ == IntervalStatus::UpdateIntervalEnd) {
    recordImpl_->setSafely(validityInterval_);
  }
  if (newEventSetupImpl && intervalStatus_ != IntervalStatus::Invalid) {
    eventSetupImpl_->addRecordImpl(*recordImpl_);
  }
}

void
EventSetupRecordProvider::endIOV(unsigned int iovIndex) {
  recordImpls_[iovIndex]->invalidateProxies();
}

void
EventSetupRecordProvider::initializeForNewSyncValue() {
  intervalStatus_ = IntervalStatus::NotInitializedForSyncValue;
}

bool EventSetupRecordProvider::legacyESSourceOutOfValidityInterval(IOVSyncValue const& iTime) const {
   if (!hasLegacyESSource_) {
      return false;
   }
   if (intervalStatus_ == IntervalStatus::Invalid ||
       !validityInterval_.validFor(iTime)) {
      return finder_->legacyOutOfValidityInterval(key_, iTime);
   }
   return false;
}

bool
EventSetupRecordProvider::setValidityIntervalFor(const IOVSyncValue& iTime)
{
   // This function can be called multiple times for the same
   // IOVSyncValue because DependentRecordIntervalFinder::setIntervalFor
   // can call it in addition to it being called directly. Don't
   // need to do the work multiple times for the same call to
   // eventSetupForInstance.
   if (intervalStatus_ == IntervalStatus::NotInitializedForSyncValue) {

      intervalStatus_ = IntervalStatus::Invalid;

      if (validityInterval_.first() != IOVSyncValue::invalidIOVSyncValue() &&
          validityInterval_.validFor(iTime)) {
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
   }
   return intervalStatus_ != IntervalStatus::Invalid;
}

void
EventSetupRecordProvider::resetProxies()
{
   // Clear out all the DataProxy's
   for (auto& recordImplIter : recordImpls_) {
      recordImplIter->invalidateProxies();
      recordImplIter->resetIfTransientInProxies();
   }
   // Force a new IOV to start with a new cacheIdentifier
   // on the next eventSetupForInstance call.
   validityInterval_ = ValidityInterval{};
   if (finder_.get() != nullptr) {
      finder_->resetInterval(key_);
   }
}

void
EventSetupRecordProvider::getReferencedESProducers(std::map<EventSetupRecordKey, std::vector<ComponentDescription const*> >& referencedESProducers) const {
   firstRecordImpl().getESProducers(referencedESProducers[key_]);
}

void
EventSetupRecordProvider::fillReferencedDataKeys(std::map<DataKey, ComponentDescription const*>& referencedDataKeys) const {
   std::vector<DataKey> keys;
   firstRecordImpl().fillRegisteredDataKeys(keys);
   std::vector<ComponentDescription const*> components = firstRecordImpl().componentsForRegisteredDataKeys();
   auto itComponents = components.begin();
   for(auto const& k: keys) {
     referencedDataKeys.emplace(k, *itComponents);
     ++itComponents;
   }
}

void
EventSetupRecordProvider::resetRecordToProxyPointers(DataToPreferredProviderMap const& iMap) {
   for (auto& recordImplIter : recordImpls_) {
      recordImplIter->clearProxies();
   }
   using std::placeholders::_1;
   for_all(providers_, std::bind(&EventSetupRecordProvider::addProxiesToRecordHelper, this, _1, iMap));
}

std::set<EventSetupRecordKey> 
EventSetupRecordProvider::dependentRecords() const
{
  return dependencies(key());
}

std::set<ComponentDescription> 
EventSetupRecordProvider::proxyProviderDescriptions() const
{
   using std::placeholders::_1;
   std::set<ComponentDescription> descriptions;
   std::transform(providers_.begin(), providers_.end(),
                  std::inserter(descriptions,descriptions.end()),
                  std::bind(&DataProxyProvider::description,_1));
   return descriptions;
}

std::shared_ptr<DataProxyProvider> 
EventSetupRecordProvider::proxyProvider(ComponentDescription const& iDesc) {
   using std::placeholders::_1;
   auto itFound = std::find_if(providers_.begin(),providers_.end(),
                std::bind(std::equal_to<ComponentDescription>(), 
                            iDesc, 
                            std::bind(&DataProxyProvider::description,_1)));
   if(itFound == providers_.end()){
      return std::shared_ptr<DataProxyProvider>();
   }
   return get_underlying_safe(*itFound);
}

std::shared_ptr<DataProxyProvider> 
EventSetupRecordProvider::proxyProvider(ParameterSetIDHolder const& psetID) {
   for (auto& dataProxyProvider : providers_) {
      if (dataProxyProvider->description().pid_ == psetID.psetID()) {
         return get_underlying_safe(dataProxyProvider);
      }
   }
   return std::shared_ptr<DataProxyProvider>();
}

void
EventSetupRecordProvider::resetProxyProvider(ParameterSetIDHolder const& psetID, std::shared_ptr<DataProxyProvider> const& sharedDataProxyProvider) {
   for (auto& dataProxyProvider : providers_) {
      if (dataProxyProvider->description().pid_ == psetID.psetID()) {
         dataProxyProvider = sharedDataProxyProvider;
         dataProxyProvider->resizeKeyedProxiesVector(key_, nConcurrentIOVs_);
      }
   }
}

std::vector<DataKey>
EventSetupRecordProvider::registeredDataKeys() const {
  std::vector<DataKey> ret;
  firstRecordImpl().fillRegisteredDataKeys(ret);
  return ret;
}

std::vector<ComponentDescription const*>
EventSetupRecordProvider::componentsForRegisteredDataKeys() const {
  return firstRecordImpl().componentsForRegisteredDataKeys();
}

   }
}
