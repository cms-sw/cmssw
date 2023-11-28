#ifndef Framework_DummyProvider_h
#define Framework_DummyProvider_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DummyProvider
//
/**\class DummyProvider DummyProvider.h FWCore/Framework/interface/DummyProvider.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu May 26 13:37:48 EDT 2005
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/DummyData.h"

#include "FWCore/Framework/interface/ESProductResolverTemplate.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"

// forward declarations
namespace edm::eventsetup::test {
  class WorkingDummyResolver : public edm::eventsetup::ESProductResolverTemplate<DummyRecord, DummyData> {
  public:
    WorkingDummyResolver(const DummyData* iDummy) : data_(iDummy) {}

  protected:
    const value_type* make(const record_type&, const DataKey&) final { return data_; }
    void const* getAfterPrefetchImpl() const final { return data_; }

  private:
    const DummyData* data_;
  };

  class DummyESProductResolverProvider : public edm::eventsetup::ESProductResolverProvider {
  public:
    DummyESProductResolverProvider(const DummyData& iData = DummyData()) : dummy_(iData) { usingRecord<DummyRecord>(); }

    void incrementData() { ++dummy_.value_; }

  protected:
    KeyedResolversVector registerResolvers(const EventSetupRecordKey&, unsigned int /* iovIndex */) override {
      KeyedResolversVector keyedResolversVector;
      edm::eventsetup::DataKey dataKey(edm::eventsetup::DataKey::makeTypeTag<DummyData>(), "");
      std::shared_ptr<WorkingDummyResolver> pResolver = std::make_shared<WorkingDummyResolver>(&dummy_);
      keyedResolversVector.emplace_back(dataKey, pResolver);
      return keyedResolversVector;
    }

  private:
    DummyData dummy_;
  };

}  // namespace edm::eventsetup::test
#endif
