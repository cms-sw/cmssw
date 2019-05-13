// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESRecordsToProxyIndices
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
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"

//
// constants, enums and typedefs
//

namespace edm::eventsetup {
  ESRecordsToProxyIndices::ESRecordsToProxyIndices(std::vector<EventSetupRecordKey> iRecords)
      : recordKeys_{std::move(iRecords)} {
    assert(std::is_sorted(iRecords.begin(), iRecords.end()));
    recordOffsets_.reserve(recordKeys_.size() + 1);
    recordOffsets_.push_back(0);
  }

  unsigned int ESRecordsToProxyIndices::dataKeysInRecord(unsigned int iRecordIndex,
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
  ESProxyIndex ESRecordsToProxyIndices::indexInRecord(EventSetupRecordKey const& iRK, DataKey const& iDK) const
      noexcept {
    auto it = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), iRK);
    if (it == recordKeys_.end() or *it != iRK) {
      return missingProxyIndex();
    }

    auto beginOffset = recordOffsets_[std::distance(recordKeys_.begin(), it)];
    ++it;
    auto endOffset = recordOffsets_[std::distance(recordKeys_.begin(), it)];

    auto itDK = std::lower_bound(dataKeys_.begin() + beginOffset, dataKeys_.begin() + endOffset, iDK);
    if (itDK == dataKeys_.begin() + endOffset or *itDK != iDK) {
      return missingProxyIndex();
    }

    return ESProxyIndex{static_cast<int>(std::distance(dataKeys_.begin() + beginOffset, itDK))};
  }

  ESRecordIndex ESRecordsToProxyIndices::recordIndexFor(EventSetupRecordKey const& iRK) const noexcept {
    auto it = std::lower_bound(recordKeys_.begin(), recordKeys_.end(), iRK);
    if (it == recordKeys_.end() or *it != iRK) {
      return missingRecordIndex();
    }
    return ESRecordIndex{static_cast<ESRecordIndex::Value_t>(it - recordKeys_.begin())};
  }

  ComponentDescription const* ESRecordsToProxyIndices::component(EventSetupRecordKey const& iRK,
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

}  // namespace edm::eventsetup
