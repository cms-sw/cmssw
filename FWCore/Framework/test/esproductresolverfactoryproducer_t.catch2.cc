/*
 *  esproductresolverfactoryproducer_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/8/05.
 *  Changed by Viji Sundararajan on 03-Jul-05.

*/

#include "FWCore/Framework/interface/ESProductResolverTemplate.h"
#include "FWCore/Framework/interface/ESProductResolverFactoryProducer.h"
#include "FWCore/Framework/test/ESProductResolverFactoryTemplate.h"
#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "catch2/catch_all.hpp"

#include <memory>
#include <string>

using namespace edm::eventsetup;
using edm::ESProductResolverFactoryProducer;
using edm::eventsetup::test::DummyData;

class DummyResolver : public ESProductResolverTemplate<DummyRecord, DummyData> {
public:
  DummyResolver() {}

protected:
  const value_type* make(const record_type&, const DataKey&) override {
    return static_cast<const value_type*>(nullptr);
  }
  void invalidateCache() override {}
  void const* getAfterPrefetchImpl() const override { return nullptr; }
};

class Test1Producer : public ESProductResolverFactoryProducer {
public:
  Test1Producer() { registerFactory(std::make_unique<ESProductResolverFactoryTemplate<DummyResolver>>()); }
};

class TestLabelProducer : public ESProductResolverFactoryProducer {
public:
  TestLabelProducer() { registerFactory(std::make_unique<ESProductResolverFactoryTemplate<DummyResolver>>(), "fred"); }
};

TEST_CASE("ESProductResolverFactoryProducer", "[Framework][EventSetup]") {
  SECTION("registerResolverfactorytemplateTest") {
    Test1Producer testProd;
    EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
    testProd.createKeyedResolvers(dummyRecordKey, 1);
    REQUIRE(testProd.isUsingRecord(dummyRecordKey));

    ESProductResolverProvider::KeyedResolvers& keyedResolvers = testProd.keyedResolvers(dummyRecordKey);

    REQUIRE(keyedResolvers.size() == 1);
    REQUIRE(nullptr != dynamic_cast<DummyResolver const*>(&(*(keyedResolvers.begin().productResolver()))));
  }

  SECTION("labelTest") {
    TestLabelProducer testProd;
    EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
    testProd.createKeyedResolvers(dummyRecordKey, 1);
    REQUIRE(testProd.isUsingRecord(dummyRecordKey));

    ESProductResolverProvider::KeyedResolvers& keyedResolvers = testProd.keyedResolvers(dummyRecordKey);

    REQUIRE(keyedResolvers.size() == 1);
    REQUIRE(nullptr != dynamic_cast<DummyResolver const*>(&(*(keyedResolvers.begin().productResolver()))));
    const std::string kFred("fred");
    REQUIRE(kFred == keyedResolvers.begin().dataKey().name().value());
  }

  SECTION("appendLabelTest") {
    edm::ParameterSet pset;
    std::string kToAppend("Barney");
    pset.addParameter("appendToDataLabel", kToAppend);
    pset.registerIt();
    {
      TestLabelProducer testProd;
      testProd.setAppendToDataLabel(pset);
      EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
      testProd.createKeyedResolvers(dummyRecordKey, 1);
      REQUIRE(testProd.isUsingRecord(dummyRecordKey));

      ESProductResolverProvider::KeyedResolvers& keyedResolvers = testProd.keyedResolvers(dummyRecordKey);

      REQUIRE(keyedResolvers.size() == 1);
      REQUIRE(nullptr != dynamic_cast<DummyResolver const*>(&(*(keyedResolvers.begin().productResolver()))));
      const std::string kFredBarney("fredBarney");
      REQUIRE(kFredBarney == keyedResolvers.begin().dataKey().name().value());
    }

    Test1Producer testProd;
    testProd.setAppendToDataLabel(pset);
    EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
    testProd.createKeyedResolvers(dummyRecordKey, 1);
    REQUIRE(testProd.isUsingRecord(dummyRecordKey));

    ESProductResolverProvider::KeyedResolvers& keyedResolvers = testProd.keyedResolvers(dummyRecordKey);

    REQUIRE(keyedResolvers.size() == 1);
    REQUIRE(nullptr != dynamic_cast<DummyResolver const*>(&(*(keyedResolvers.begin().productResolver()))));
    const std::string kBarney("Barney");
    REQUIRE(kBarney == keyedResolvers.begin().dataKey().name().value());
  }
}
