#ifndef FWCore_Framework_ModuleConsumesMinimalESInfo_h
#define FWCore_Framework_ModuleConsumesMinimalESInfo_h

// -*- C++ -*-
// Package:     FWCore/Framework
// Class  :     ModuleConsumesMinimalESInfo
// Minimal information about the consumes call for an EventSetup product
// requested from an ED module

#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataKey.h"
#include <string_view>

namespace edm {
  struct ModuleConsumesMinimalESInfo {
    ModuleConsumesMinimalESInfo(Transition iTransition,
                                eventsetup::EventSetupRecordKey const& iRecord,
                                eventsetup::DataKey const& iDataKey,
                                std::string_view iComponentLabel,
                                ESResolverIndex iIndex)
        : transition_(iTransition),
          record_(iRecord),
          dataKey_(iDataKey.type(), iDataKey.name(), eventsetup::DataKey::DoNotCopyMemory()),
          componentLabel_(iComponentLabel) {}
    ModuleConsumesMinimalESInfo() = default;
    ModuleConsumesMinimalESInfo(ModuleConsumesMinimalESInfo&&) = default;
    ModuleConsumesMinimalESInfo& operator=(ModuleConsumesMinimalESInfo&&) = default;

    //want to avoid accidently copying dataKey_
    ModuleConsumesMinimalESInfo(ModuleConsumesMinimalESInfo const&) = delete;
    ModuleConsumesMinimalESInfo& operator=(ModuleConsumesMinimalESInfo const&) = delete;

    edm::Transition transition_;
    eventsetup::EventSetupRecordKey record_;
    eventsetup::DataKey dataKey_;
    std::string_view componentLabel_;
  };
}  // namespace edm
#endif  // FWCore_Framework_ModuleConsumesMinimalESInfo_h