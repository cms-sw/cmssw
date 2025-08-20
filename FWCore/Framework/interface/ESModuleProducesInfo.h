#ifndef FWCore_Framework_ESModuleProducesInfo_h
#define FWCore_Framework_ESModuleProducesInfo_h
// -*- C++ -*-

// Package:     Framework
// Class  :     ESModuleProducesInfo
//
// Description: Contains information about which products
// a module declares it will produce from the EventSetup.

#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
namespace edm::eventsetup {
  class ESModuleProducesInfo {
  public:
    ESModuleProducesInfo(EventSetupRecordKey const& iRecord, DataKey const& iDataKey, unsigned int iProduceMethodID)
        : record_(iRecord), dataKey_(iDataKey), produceMethodID_(iProduceMethodID) {}

    EventSetupRecordKey const& record() const { return record_; }
    DataKey const& dataKey() const { return dataKey_; }
    unsigned int produceMethodID() const { return produceMethodID_; }

  private:
    EventSetupRecordKey record_;
    DataKey dataKey_;
    unsigned int produceMethodID_;
  };
}  // namespace edm::eventsetup
#endif