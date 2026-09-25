#ifndef Framework_EventSetupRecordInfiniteIntervalFinder_h
#define Framework_EventSetupRecordInfiniteIntervalFinder_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordInfiniteIntervalFinder
//
/**\class EventSetupRecordInfiniteIntervalFinder EventSetupRecordInfiniteIntervalFinder.h FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Tue Mar 29 16:15:11 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

// forward declarations
namespace edm {

  class EventSetupRecordInfiniteIntervalFinder : public EventSetupRecordIntervalFinder {
  public:
    EventSetupRecordInfiniteIntervalFinder() = default;

    // ---------- member functions ---------------------------
  private:
    void setIntervalFor(const eventsetup::EventSetupRecordKey&, const IOVSyncValue&, ValidityInterval& oIOV) final {
      oIOV = edm::ValidityInterval(IOVSyncValue::beginOfTime(), IOVSyncValue::endOfTime());
    }
  };

}  // namespace edm
#endif
