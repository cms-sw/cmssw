#ifndef FWCore_Framework_ESModuleConsumesMinimalInfo_h
#define FWCore_Framework_ESModuleConsumesMinimalInfo_h

// -*- C++ -*-
// Package:     FWCore/Framework
// Class  :     ESModuleConsumesMinimalInfo
// Minimal information about the consumes call for an EventSetup product by an ESModule

#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataKey.h"
#include <string_view>

namespace edm::eventsetup {
  struct ESModuleConsumesMinimalInfo {
    ESModuleConsumesMinimalInfo(unsigned int iProduceMethodID,
                                eventsetup::EventSetupRecordKey const& iRecord,
                                eventsetup::DataKey const& iDataKey,
                                std::string_view iComponentLabel)
        : recordForDataKey_(iRecord),
          dataKey_(iDataKey.type(), iDataKey.name(), eventsetup::DataKey::DoNotCopyMemory()),
          componentLabel_(iComponentLabel),
          produceMethodID_(iProduceMethodID) {}
    ESModuleConsumesMinimalInfo() = default;
    ESModuleConsumesMinimalInfo(ESModuleConsumesMinimalInfo&&) = default;
    ESModuleConsumesMinimalInfo& operator=(ESModuleConsumesMinimalInfo&&) = default;

    // avoid accidentally copying data key as the strings would be copied instead of shared
    ESModuleConsumesMinimalInfo(ESModuleConsumesMinimalInfo const&) = delete;
    ESModuleConsumesMinimalInfo& operator=(ESModuleConsumesMinimalInfo const&) = delete;

    eventsetup::EventSetupRecordKey recordForDataKey_;
    eventsetup::DataKey dataKey_;
    std::string_view componentLabel_;
    unsigned int produceMethodID_ = 0;
  };
}  // namespace edm::eventsetup
#endif  // FWCore_Framework_ESModuleConsumesMinimalInfo_h