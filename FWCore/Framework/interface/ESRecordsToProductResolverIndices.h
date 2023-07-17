#ifndef FWCore_Framework_ESRecordsToProductResolverIndices_h
#define FWCore_Framework_ESRecordsToProductResolverIndices_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESRecordsToProductResolverIndices
//
/**\class ESRecordsToProductResolverIndices ESRecordsToProductResolverIndices.h FWCore/Framework/interface/ESRecordsToProductResolverIndices.h

 Description: Key used to identify data within a EventSetupRecord

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Tues Feb 12 14:31:03 EST 2019
//

// system include files
#include <limits>
#include <vector>

// user include files
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ESTagGetter.h"
#include "FWCore/Utilities/interface/ESIndices.h"

// forward declarations
namespace edm::eventsetup {
  struct ComponentDescription;

  class ESRecordsToProductResolverIndices {
  public:
    ESRecordsToProductResolverIndices(std::vector<EventSetupRecordKey> iRecords);

    // ---------- const member functions ---------------------
    ///If the index is not found, returns missingResolverIndex()
    ESResolverIndex indexInRecord(EventSetupRecordKey const& iRK, DataKey const& iDK) const noexcept;

    ComponentDescription const* component(EventSetupRecordKey const& iRK, DataKey const& iDK) const noexcept;

    ///Returns ESTagGetter for all products matching the type iTT for record iRK
    ESTagGetter makeTagGetter(EventSetupRecordKey const& iRK, TypeTag const& iTT) const;

    static constexpr ESResolverIndex missingResolverIndex() noexcept {
      return ESResolverIndex{std::numeric_limits<int>::max()};
    }
    static constexpr ESRecordIndex missingRecordIndex() noexcept {
      return ESRecordIndex{ESRecordIndex::invalidValue()};
    }

    ESRecordIndex recordIndexFor(EventSetupRecordKey const& iRK) const noexcept;

    std::pair<std::vector<DataKey>::const_iterator, std::vector<DataKey>::const_iterator> keysForRecord(
        EventSetupRecordKey const& iRK) const noexcept;
    ///The sorted list of keys
    std::vector<EventSetupRecordKey> recordKeys() const noexcept { return recordKeys_; }

    // ---------- member functions ---------------------------
    ///This should be called for all records in the list passed to the constructor and
    /// in the same order as the list.
    unsigned int dataKeysInRecord(unsigned int iRecordIndex,
                                  EventSetupRecordKey const& iRecord,
                                  std::vector<DataKey> const& iDataKeys,
                                  std::vector<ComponentDescription const*> const& iComponents);

  private:
    // ---------- member data --------------------------------
    std::vector<EventSetupRecordKey> recordKeys_;
    //for each item in recordKeys_ this holds the index to the first
    // DataKey for that item in dataKeys_
    // recordOffset_ is 1 size larger than recordKeys_ with the
    // last entry being dataKeys_.size()
    std::vector<unsigned int> recordOffsets_;
    std::vector<DataKey> dataKeys_;
    std::vector<ComponentDescription const*> components_;
  };

}  // namespace edm::eventsetup
#endif
