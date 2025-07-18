// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     EventSetupTestHelper
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  root
//         Created:  Tue, 08 May 2018 18:33:15 GMT
//

// system include files

// user include files
#include "FWCore/TestProcessor/interface/EventSetupTestHelper.h"
#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Framework/interface/ESModuleProducesInfo.h"

namespace edm {
  namespace test {

    EventSetupTestHelper::EventSetupTestHelper(std::vector<ESProduceEntry> iResolvers)
        : resolvers_{std::move(iResolvers)} {
      //Deal with duplicates
      std::set<eventsetup::EventSetupRecordKey> records;
      for (auto const& p : resolvers_) {
        records.insert(p.recordKey_);
      }
      for (auto const& k : records) {
        usingRecordWithKey(k);
        findingRecordWithKey(k);
      }
    }

    void EventSetupTestHelper::setIntervalFor(const eventsetup::EventSetupRecordKey&,
                                              const IOVSyncValue& iSync,
                                              ValidityInterval& oIOV) {
      // Note that we manually invalidate the resolvers at the end of every call
      // to test. And the beginning of the call to test is the only opportunity
      // to reset this data, so we are not relying on the EventSetup system
      // to manage invalidating the resolvers in EventSetupTestHelper. The only
      // reasonable thing to do is return an interval for all time so the EventSetup
      // system does not invalidate these resolvers when it shouldn't. There are two
      // weaknesses to this:
      //
      //     1. If for the same record type there are ESProductResolvers both managed
      //     by this class and also others managed by the EventSetup, then
      //     at IOV boundaries for this record this will fail. The EventSetup
      //     will invalidate all the resolvers for the record after this class
      //     has set the ones it manages and they will stay invalid when they
      //     are needed.
      //
      //     2. TestProcessor does not support the special case where the different
      //     transitions executed in one call to test have different IOVs and different
      //     EventSetup data. That would be a pretty strange case, especially for a test.

      oIOV = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
    }

    eventsetup::ESProductResolverProvider::KeyedResolversVector EventSetupTestHelper::registerResolvers(
        const eventsetup::EventSetupRecordKey& iRecordKey, unsigned int iovIndex) {
      KeyedResolversVector keyedResolversVector;
      for (auto const& p : resolvers_) {
        if (p.recordKey_ == iRecordKey) {
          keyedResolversVector.emplace_back(p.dataKey_, p.resolver_);
        }
      }
      return keyedResolversVector;
    }

    std::shared_ptr<eventsetup::ESProductResolver> EventSetupTestHelper::getResolver(unsigned int iIndex) {
      return resolvers_[iIndex].resolver_;
    }

    std::vector<eventsetup::ESModuleProducesInfo> EventSetupTestHelper::producesInfo() const {
      std::vector<eventsetup::ESModuleProducesInfo> producesInfo;
      producesInfo.reserve(resolvers_.size());
      for (auto const& p : resolvers_) {
        producesInfo.emplace_back(p.recordKey_, p.dataKey_, p.resolver_->produceMethodID());
      }
      return producesInfo;
    }

    void EventSetupTestHelper::resetAllResolvers() {
      for (auto const& p : resolvers_) {
        p.resolver_->invalidate();
      }
    }

  }  // namespace test
}  // namespace edm
