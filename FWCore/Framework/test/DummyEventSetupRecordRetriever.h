#ifndef Framework_DummyEventSetupRecordRetriever_h
#define Framework_DummyEventSetupRecordRetriever_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DummyEventSetupRecordRetriever
//
/**\class DummyEventSetupRecordRetriever DummyEventSetupRecordRetriever.h FWCore/Framework/interface/DummyEventSetupRecordRetriever.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr 22 14:14:09 EDT 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/test/DummyEventSetupRecord.h"
#include "FWCore/Framework/test/DummyEventSetupData.h"

// forward declarations
namespace edm {
  class DummyEventSetupRecordRetriever : public EventSetupRecordIntervalFinder, public ESProducer {
  public:
    DummyEventSetupRecordRetriever() {
      this->findingRecord<DummyEventSetupRecord>();
      setWhatProduced(this);
    }

    std::unique_ptr<DummyEventSetupData> produce(const DummyEventSetupRecord&) {
      return std::make_unique<DummyEventSetupData>(1);
    }

  protected:
    virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                const edm::IOVSyncValue& /*iTime*/,
                                edm::ValidityInterval& iInterval) {
      iInterval = edm::ValidityInterval(IOVSyncValue::beginOfTime(), IOVSyncValue::endOfTime());
    }

  private:
    DummyEventSetupRecordRetriever(const DummyEventSetupRecordRetriever&);  // stop default

    const DummyEventSetupRecordRetriever& operator=(const DummyEventSetupRecordRetriever&);  // stop default

    // ---------- member data --------------------------------
  };
}  // namespace edm
#endif
