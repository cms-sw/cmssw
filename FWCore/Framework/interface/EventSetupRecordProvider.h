// -*- C++ -*-
#ifndef FWCore_Framework_EventSetupRecordProvider_h
#define FWCore_Framework_EventSetupRecordProvider_h
//
// Package:     Framework
// Class  :     EventSetupRecordProvider
//
/**\class edm::eventsetup::EventSetupRecordProvider

 Description: Coordinates all EventSetupDataProviders with the same 'interval of validity'

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Mar 25 19:02:23 EST 2005
//

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// system include files
#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

// forward declarations
namespace edm {
  class ActivityRegistry;
  class EventSetupRecordIntervalFinder;

  namespace eventsetup {
    struct ComponentDescription;
    class DataKey;
    class ESProductResolverProvider;
    class ESRecordsToProductResolverIndices;
    class EventSetupRecordImpl;

    class EventSetupRecordProvider {
    public:
      enum class IntervalStatus { NotInitializedForSyncValue, Invalid, NewInterval, UpdateIntervalEnd, SameInterval };

      using DataToPreferredProviderMap = std::map<DataKey, ComponentDescription>;

      EventSetupRecordProvider(EventSetupRecordKey const& iKey,
                               ActivityRegistry const*,
                               unsigned int nConcurrentIOVs = 1);

      EventSetupRecordProvider(EventSetupRecordProvider const&) = delete;
      EventSetupRecordProvider const& operator=(EventSetupRecordProvider const&) = delete;

      // ---------- const member functions ---------------------

      unsigned int nConcurrentIOVs() const { return nConcurrentIOVs_; }

      EventSetupRecordKey const& key() const { return key_; }

      // Returns a reference for the first of the allowed concurrent IOVs.
      // There is always at least one IOV allowed. Intended for cases where
      // you only need one of them and don't care which one you get.
      EventSetupRecordImpl const& firstRecordImpl() const;

      ///Returns the list of Records the provided Record depends on (usually none)
      std::set<EventSetupRecordKey> supportingRecords() const;

      ///return information on which ESProductResolverProviders are supplying information
      std::set<ComponentDescription> resolverProviderDescriptions() const;

      /// The available DataKeys in the Record. The order can be used to request the data by index
      std::vector<DataKey> const& registeredDataKeys() const;

      std::vector<ComponentDescription const*> componentsForRegisteredDataKeys() const;
      std::vector<unsigned int> produceMethodIDsForRegisteredDataKeys() const;

      // ---------- member functions ---------------------------

      ///returns the first matching ESProductResolverProvider or a 'null' if not found
      std::shared_ptr<ESProductResolverProvider> resolverProvider(ComponentDescription const&);

      void add(std::shared_ptr<ESProductResolverProvider>);

      /** This function is called when an IOV is started up by the asynchronous task assigned to start IOVs.
       *  This is the point where we know the value of iovIndex, so we know which EventSetupRecordImpl
       *  object will be used. In the EventSetupRecordImpl we set the cacheIdentifier
       *  and validity interval.
       */
      EventSetupRecordImpl const& initializeForNewIOV(unsigned int iovIndex,
                                                      unsigned long long cacheIdentifier,
                                                      ValidityInterval const& validityInterval);

      /** This function is called when we do not need to start a new IOV for a record and syncValue.
       *  This is also the place where the validity interval will be updated in the special case
       *  where the beginning of the interval did not change but the end changed so it is not treated
       *  as a new IOV.
       */
      EventSetupRecordImpl const& continueIOV(unsigned int iovIndex, edm::ValidityInterval const* validityInterval);

      /** The asynchronous task called when an IOV ends calls this function. It clears the caches
       *  of the ESProductResolver's.
       */
      void endIOV(unsigned int iovIndex);

      /**In the case of a conflict, sets what Provider to call.  This must be called after
         all providers have been added.  An empty map is acceptable. */
      void usePreferred(DataToPreferredProviderMap const&);

      ///This will clear the cache's of all the Resolvers so that next time they are called they will run
      void resetResolvers();

      void getReferencedESProducers(
          std::map<EventSetupRecordKey, std::vector<ComponentDescription const*>>& referencedESProducers) const;

      void fillAllESProductResolverProviders(std::vector<ESProductResolverProvider const*>&,
                                             std::unordered_set<unsigned int>& componentIDs) const;

      void updateLookup(ESRecordsToProductResolverIndices const&);

    protected:
      void addResolversToRecordHelper(edm::propagate_const<std::shared_ptr<ESProductResolverProvider>>& dpp,
                                      DataToPreferredProviderMap const& mp) {
        addResolversToRecord(get_underlying_safe(dpp), mp);
      }
      void addResolversToRecord(std::shared_ptr<ESProductResolverProvider>, DataToPreferredProviderMap const&);

    private:
      // ---------- member data --------------------------------
      const EventSetupRecordKey key_;

      std::vector<EventSetupRecordImpl> recordImpls_;

      std::vector<edm::propagate_const<std::shared_ptr<ESProductResolverProvider>>> providers_;

      const unsigned int nConcurrentIOVs_;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
