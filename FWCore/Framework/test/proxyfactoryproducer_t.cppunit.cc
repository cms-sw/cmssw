/*
 *  proxyfactoryproducer_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/8/05.
 *  Changed by Viji Sundararajan on 03-Jul-05.

*/

#include "FWCore/Framework/interface/DataProxyTemplate.h"
#include "FWCore/Framework/interface/ESProxyFactoryProducer.h"
#include "FWCore/Framework/interface/ProxyFactoryTemplate.h"
#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "cppunit/extensions/HelperMacros.h"

#include <memory>
#include <string>

using namespace edm::eventsetup;
using edm::ESProxyFactoryProducer;
using edm::eventsetup::test::DummyData;

class DummyProxy : public DataProxyTemplate<DummyRecord, DummyData> {
public:
  DummyProxy() {}

protected:
  const value_type* make(const record_type&, const DataKey&) { return static_cast<const value_type*>(nullptr); }
  void invalidateCache() {}
};

class Test1Producer : public ESProxyFactoryProducer {
public:
  Test1Producer() { registerFactory(std::make_unique<ProxyFactoryTemplate<DummyProxy>>()); }
};

class TestLabelProducer : public ESProxyFactoryProducer {
public:
  TestLabelProducer() { registerFactory(std::make_unique<ProxyFactoryTemplate<DummyProxy>>(), "fred"); }
};

class testProxyfactor : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testProxyfactor);

  CPPUNIT_TEST(registerProxyfactorytemplateTest);
  CPPUNIT_TEST(labelTest);
  CPPUNIT_TEST(appendLabelTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void registerProxyfactorytemplateTest();
  void appendLabelTest();
  void labelTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testProxyfactor);

void testProxyfactor::registerProxyfactorytemplateTest() {
  Test1Producer testProd;
  EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
  testProd.createKeyedProxies(dummyRecordKey, 1);
  CPPUNIT_ASSERT(testProd.isUsingRecord(dummyRecordKey));

  DataProxyProvider::KeyedProxies& keyedProxies = testProd.keyedProxies(dummyRecordKey);

  CPPUNIT_ASSERT(keyedProxies.size() == 1);
  CPPUNIT_ASSERT(nullptr != dynamic_cast<DummyProxy const*>(&(*(keyedProxies.begin().dataProxy()))));
}

void testProxyfactor::labelTest() {
  TestLabelProducer testProd;
  EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
  testProd.createKeyedProxies(dummyRecordKey, 1);
  CPPUNIT_ASSERT(testProd.isUsingRecord(dummyRecordKey));

  DataProxyProvider::KeyedProxies& keyedProxies = testProd.keyedProxies(dummyRecordKey);

  CPPUNIT_ASSERT(keyedProxies.size() == 1);
  CPPUNIT_ASSERT(nullptr != dynamic_cast<DummyProxy const*>(&(*(keyedProxies.begin().dataProxy()))));
  const std::string kFred("fred");
  CPPUNIT_ASSERT(kFred == keyedProxies.begin().dataKey().name().value());
}

void testProxyfactor::appendLabelTest() {
  edm::ParameterSet pset;
  std::string kToAppend("Barney");
  pset.addParameter("appendToDataLabel", kToAppend);
  pset.registerIt();
  {
    TestLabelProducer testProd;
    testProd.setAppendToDataLabel(pset);
    EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
    testProd.createKeyedProxies(dummyRecordKey, 1);
    CPPUNIT_ASSERT(testProd.isUsingRecord(dummyRecordKey));

    DataProxyProvider::KeyedProxies& keyedProxies = testProd.keyedProxies(dummyRecordKey);

    CPPUNIT_ASSERT(keyedProxies.size() == 1);
    CPPUNIT_ASSERT(nullptr != dynamic_cast<DummyProxy const*>(&(*(keyedProxies.begin().dataProxy()))));
    const std::string kFredBarney("fredBarney");
    CPPUNIT_ASSERT(kFredBarney == keyedProxies.begin().dataKey().name().value());
  }

  Test1Producer testProd;
  testProd.setAppendToDataLabel(pset);
  EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
  testProd.createKeyedProxies(dummyRecordKey, 1);
  CPPUNIT_ASSERT(testProd.isUsingRecord(dummyRecordKey));

  DataProxyProvider::KeyedProxies& keyedProxies = testProd.keyedProxies(dummyRecordKey);

  CPPUNIT_ASSERT(keyedProxies.size() == 1);
  CPPUNIT_ASSERT(nullptr != dynamic_cast<DummyProxy const*>(&(*(keyedProxies.begin().dataProxy()))));
  const std::string kBarney("Barney");
  CPPUNIT_ASSERT(kBarney == keyedProxies.begin().dataKey().name().value());
}
