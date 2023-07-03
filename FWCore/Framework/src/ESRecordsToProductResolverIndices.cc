// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESRecordsToProductResolverIndices
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 31 14:31:13 EST 2005
//

// system include files
#include <cassert>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

//
// constants, enums and typedefs
//

namespace edm::eventsetup {
  ESRecordsToProductResolverIndices::ESRecordsToProductResolverIndices(std::vector<EventSetupRecordKey> iRecords)
      : recordKeys_{std::move(iRecords)} {
    assert(std::is_sorted(recordKeys_.begin(), recordKeys_.end()));
    recordOffsets_.reserve(recordKeys_.size() + 1);
    recordOffsets_.push_back(0);
  }

  unsigned int ESRecordsToProductResolverIndices::dataKeysInRecord(
      unsigned int iRecordIndex,
      EventSetupRecordKey const& iRecord,
      std::vector<DataKey> const& iDataKeys,
      std::vector<ComponentDescription const*> const& iComponents) {
    assert(iRecord == recordKeys_[iRecordIndex]);
    assert(iDataKeys.size() == iComponents.size());
    assert(iRecordIndex + 1 == recordOffsets_.size());
    dataKeys_.insert(dataKeys_.end(), iDataKeys.begin(), iDataKeys.end());
    ++iRecordIndex;
    components_.insert(components_.end(), iComponents.begin(), iComponents.end());
    recordOffsets_.push_back(dataKeys_.size());
    return iRecordIndex;
  }

  //
  // const member functions
  //
  ESResolverIndex ESRecordsToProductResolverIndices::indexInRecord(EventSetupRecordKey const& iRK,
                                                                   DataKey const& iDK) const noexcept {
    auto it = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), iRK);
    if (it == recordKeys_.end() or *it != iRK) {
      return missingResolverIndex();
    }

    auto beginOffset = recordOffsets_[std::distance(recordKeys_.begin(), it)];
    ++it;
    auto endOffset = recordOffsets_[std::distance(recordKeys_.begin(), it)];

    auto itDK = std::lower_bound(dataKeys_.begin() + beginOffset, dataKeys_.begin() + endOffset, iDK);
    if (itDK == dataKeys_.begin() + endOffset or *itDK != iDK) {
      return missingResolverIndex();
    }

    return ESResolverIndex{static_cast<int>(std::distance(dataKeys_.begin() + beginOffset, itDK))};
  }

  ESRecordIndex ESRecordsToProductResolverIndices::recordIndexFor(EventSetupRecordKey const& iRK) const noexcept {
    auto it = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), iRK);
    if (it == recordKeys_.end() or *it != iRK) {
      return missingRecordIndex();
    }
    return ESRecordIndex{static_cast<ESRecordIndex::Value_t>(it - recordKeys_.begin())};
  }

  ComponentDescription const* ESRecordsToProductResolverIndices::component(EventSetupRecordKey const& iRK,
                                                                           DataKey const& iDK) const noexcept {
    auto it = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), iRK);
    if (it == recordKeys_.end() or *it != iRK) {
      return nullptr;
    }

    auto beginOffset = recordOffsets_[std::distance(recordKeys_.begin(), it)];
    ++it;
    auto endOffset = recordOffsets_[std::distance(recordKeys_.begin(), it)];

    auto itDK = std::lower_bound(dataKeys_.begin() + beginOffset, dataKeys_.begin() + endOffset, iDK);
    if (itDK == dataKeys_.begin() + endOffset or *itDK != iDK) {
      return nullptr;
    }
    return components_[std::distance(dataKeys_.begin(), itDK)];
  }

  ESTagGetter ESRecordsToProductResolverIndices::makeTagGetter(EventSetupRecordKey const& iRK,
                                                               TypeTag const& iTT) const {
    auto recIndex = recordIndexFor(iRK);
    if (recIndex == missingRecordIndex()) {
      return ESTagGetter();
    }

    auto const beginIndex = recordOffsets_[recIndex.value()];
    auto const endIndex = recordOffsets_[recIndex.value() + 1];
    auto keyIndex = beginIndex;

    std::vector<ESTagGetter::Info> returnValue;
    returnValue.reserve(endIndex - beginIndex);
    bool foundFirstIndex = false;
    while (keyIndex < endIndex) {
      if (dataKeys_[keyIndex].type() != iTT) {
        if (foundFirstIndex) {
          //we are now past any further matches
          break;
        }
      } else {
        foundFirstIndex = true;
        returnValue.emplace_back(
            keyIndex - beginIndex,
            dataKeys_[keyIndex].name().value(),
            components_[keyIndex] ? std::string_view(components_[keyIndex]->label_) : std::string_view());
      }
      ++keyIndex;
    }
    return returnValue;
  }

  std::pair<std::vector<DataKey>::const_iterator, std::vector<DataKey>::const_iterator>
  ESRecordsToProductResolverIndices::keysForRecord(EventSetupRecordKey const& iRK) const noexcept {
    auto recIndex = recordIndexFor(iRK);
    if (recIndex == missingRecordIndex()) {
      return std::make_pair(dataKeys_.end(), dataKeys_.end());
    }
    auto const beginIndex = recordOffsets_[recIndex.value()];
    auto const endIndex = recordOffsets_[recIndex.value() + 1];
    return std::make_pair(dataKeys_.begin() + beginIndex, dataKeys_.begin() + endIndex);
  }

}  // namespace edm::eventsetup
