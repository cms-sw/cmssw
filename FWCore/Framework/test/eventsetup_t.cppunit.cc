/*
 *  eventsetup_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/24/05.
 *  Changed by Viji Sundararajan on 24-Jun-05.
 *
 */

#include <iostream>

#include "cppunit/extensions/HelperMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"


#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/DummyProxyProvider.h"

#include "FWCore/Framework/interface/HCMethods.h"

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/Framework/test/DummyEventSetupRecordRetriever.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

using namespace edm;
namespace {
  bool non_null(const void* iPtr) {
    return iPtr != nullptr;
  }
}

class testEventsetup: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testEventsetup);

  CPPUNIT_TEST(constructTest);
  CPPUNIT_TEST(getTest);
  CPPUNIT_TEST(tryToGetTest);
  CPPUNIT_TEST_EXCEPTION(getExcTest,edm::eventsetup::NoRecordException<DummyRecord>);
  CPPUNIT_TEST(recordProviderTest);
  CPPUNIT_TEST(provenanceTest);
  CPPUNIT_TEST(getDataWithLabelTest);
  CPPUNIT_TEST(getDataWithESInputTagTest);
  CPPUNIT_TEST_EXCEPTION(recordValidityTest,edm::eventsetup::NoRecordException<DummyRecord>);
  CPPUNIT_TEST_EXCEPTION(recordValidityExcTest,edm::eventsetup::NoRecordException<DummyRecord>);
  CPPUNIT_TEST(proxyProviderTest);

  CPPUNIT_TEST_EXCEPTION(producerConflictTest,cms::Exception);
  CPPUNIT_TEST_EXCEPTION(sourceConflictTest,cms::Exception);
  CPPUNIT_TEST(twoSourceTest);
  CPPUNIT_TEST(sourceProducerResolutionTest);
  CPPUNIT_TEST(preferTest);

  CPPUNIT_TEST(introspectionTest);

  CPPUNIT_TEST(iovExtentionTest);

  CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void constructTest();
  void getTest();
  void tryToGetTest();
  void getExcTest();
  void recordProviderTest();
  void recordValidityTest();
  void recordValidityExcTest();
  void proxyProviderTest();
  void provenanceTest();
  void getDataWithLabelTest();
  void getDataWithESInputTagTest();

  void producerConflictTest();
  void sourceConflictTest();
  void twoSourceTest();
  void sourceProducerResolutionTest();
  void preferTest();

  void introspectionTest();

  void iovExtentionTest();

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventsetup);

namespace {
  edm::ActivityRegistry activityRegistry;
}

void testEventsetup::constructTest()
{
  eventsetup::EventSetupProvider provider(&activityRegistry);
  const Timestamp time(1);
  const IOVSyncValue timestamp(time);
  EventSetup const& eventSetup = provider.eventSetupForInstance(timestamp);
  CPPUNIT_ASSERT(non_null(&eventSetup));
}

void testEventsetup::getTest()
{
  eventsetup::EventSetupProvider provider(&activityRegistry);
  EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
  CPPUNIT_ASSERT(non_null(&eventSetup));

  eventsetup::EventSetupRecordImpl dummyRecord{ eventsetup::EventSetupRecordKey::makeKey<DummyRecord>() };
  provider.addRecordToEventSetup(dummyRecord);
  const DummyRecord& gottenRecord = eventSetup.get<DummyRecord>();
  CPPUNIT_ASSERT(non_null(&gottenRecord));
  CPPUNIT_ASSERT(&dummyRecord == gottenRecord.impl_);
}

void testEventsetup::tryToGetTest()
{
  eventsetup::EventSetupProvider provider(&activityRegistry);
  EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
  CPPUNIT_ASSERT(non_null(&eventSetup));

  eventsetup::EventSetupRecordImpl dummyRecord{ eventsetup::EventSetupRecordKey::makeKey<DummyRecord>() };
  provider.addRecordToEventSetup(dummyRecord);
  auto gottenRecord = eventSetup.tryToGet<DummyRecord>();
  CPPUNIT_ASSERT(gottenRecord);
  CPPUNIT_ASSERT(&dummyRecord == gottenRecord->impl_);
}

void testEventsetup::getExcTest()
{
  eventsetup::EventSetupProvider provider(&activityRegistry);
  EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
  CPPUNIT_ASSERT(non_null(&eventSetup));
  eventSetup.get<DummyRecord>();
}


class DummyEventSetupProvider : public edm::eventsetup::EventSetupProvider {
public:

  DummyEventSetupProvider(ActivityRegistry* activityRegistry) : EventSetupProvider(activityRegistry) { }

  template<class T>
  void insert(std::unique_ptr<T> iRecord) {
    edm::eventsetup::EventSetupProvider::insert(std::move(iRecord));
  }
};

void testEventsetup::recordProviderTest()
{
  DummyEventSetupProvider provider(&activityRegistry);
  typedef eventsetup::EventSetupRecordProvider DummyRecordProvider;
  provider.insert(std::make_unique<DummyRecordProvider>(DummyRecord::keyForClass()));

  //NOTE: use 'invalid' timestamp since the default 'interval of validity'
  //       for a Record is presently an 'invalid' timestamp on both ends.
  //       Since the EventSetup::get<> will only retrieve a Record if its
  //       interval of validity is 'valid' for the present 'instance'
  //       this is a 'hack' to have the 'get' succeed
  EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
  const DummyRecord& gottenRecord = eventSetup.get<DummyRecord>();
  CPPUNIT_ASSERT(non_null(&gottenRecord));
}


class DummyFinder : public EventSetupRecordIntervalFinder {
public:
  DummyFinder() : EventSetupRecordIntervalFinder(), interval_()  {
    this->findingRecord<DummyRecord>();
  }

  void setInterval(const ValidityInterval& iInterval) {
    interval_ = iInterval;
  }
protected:
  virtual void setIntervalFor(const eventsetup::EventSetupRecordKey&,
                              const IOVSyncValue& iTime,
                              ValidityInterval& iInterval) {
    if(interval_.validFor(iTime)) {
      iInterval = interval_;
    } else {
      iInterval = ValidityInterval();
    }
  }
private:
  ValidityInterval interval_;
};


void testEventsetup::recordValidityTest()
{
  DummyEventSetupProvider provider(&activityRegistry);
  typedef eventsetup::EventSetupRecordProvider DummyRecordProvider;
  auto dummyRecordProvider = std::make_unique<DummyRecordProvider>(DummyRecord::keyForClass());
  auto finder = std::make_shared<DummyFinder>();
  dummyRecordProvider->addFinder(finder);

  provider.insert(std::move(dummyRecordProvider));

  {
    Timestamp time_1(1);
    provider.eventSetupForInstance(IOVSyncValue(time_1));
  }

  const Timestamp time_2(2);
  finder->setInterval(ValidityInterval(IOVSyncValue(time_2), IOVSyncValue(Timestamp(3))));
  {
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue(time_2));
    eventSetup.get<DummyRecord>();
  }
  {
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue(Timestamp(3)));
    eventSetup.get<DummyRecord>();
  }
  {
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue(Timestamp(4)));
    eventSetup.get<DummyRecord>();
  }


}

void testEventsetup::recordValidityExcTest()
{
  DummyEventSetupProvider provider(&activityRegistry);
  typedef eventsetup::EventSetupRecordProvider DummyRecordProvider;
  auto dummyRecordProvider = std::make_unique<DummyRecordProvider>(DummyRecord::keyForClass());
  dummyRecordProvider->addFinder(std::make_shared<DummyFinder>());

  provider.insert(std::move(dummyRecordProvider));

  {
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue(Timestamp(1)));
    eventSetup.get<DummyRecord>();
  }

}

class DummyProxyProvider : public eventsetup::DataProxyProvider {
public:
  DummyProxyProvider() {
    usingRecord<DummyRecord>();
  }
  void newInterval(const eventsetup::EventSetupRecordKey& /*iRecordType*/,
                   const ValidityInterval& /*iInterval*/) {
    //do nothing
  }
protected:
  void registerProxies(const eventsetup::EventSetupRecordKey&, KeyedProxies& /*iHolder*/) {
  }

};

//create an instance of the register
static eventsetup::RecordDependencyRegister<DummyRecord> s_factory;

void testEventsetup::proxyProviderTest()
{
  eventsetup::EventSetupProvider provider(&activityRegistry);
  provider.add(std::make_shared<DummyProxyProvider>());

  EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
  const DummyRecord& gottenRecord = eventSetup.get<DummyRecord>();
  CPPUNIT_ASSERT(non_null(&gottenRecord));
}

void testEventsetup::producerConflictTest()
{
  edm::eventsetup::ComponentDescription description("DummyProxyProvider","",false);
  using edm::eventsetup::test::DummyProxyProvider;
  eventsetup::EventSetupProvider provider(&activityRegistry);
  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  //checking for conflicts is now delayed until first time EventSetup is requested
  /*EventSetup const& eventSetup = */ provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());

}
void testEventsetup::sourceConflictTest()
{
  edm::eventsetup::ComponentDescription description("DummyProxyProvider","",true);
  using edm::eventsetup::test::DummyProxyProvider;
  eventsetup::EventSetupProvider provider(&activityRegistry);
  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  //checking for conflicts is now delayed until first time EventSetup is requested
  /*EventSetup const& eventSetup = */ provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());

}
//#define TEST_EXCLUDE_DEF

void testEventsetup::twoSourceTest()
{
  edm::eventsetup::ComponentDescription description("DummyProxyProvider","",true);
  using edm::eventsetup::test::DummyProxyProvider;
  eventsetup::EventSetupProvider provider(&activityRegistry);
  {
    auto dummyProv = std::make_shared<DummyProxyProvider>();
    dummyProv->setDescription(description);
    provider.add(dummyProv);
  }
  {
    auto dummyProv = std::make_shared<edm::DummyEventSetupRecordRetriever>();
    std::shared_ptr<eventsetup::DataProxyProvider> providerPtr(dummyProv);
    std::shared_ptr<edm::EventSetupRecordIntervalFinder> finderPtr(dummyProv);
    edm::eventsetup::ComponentDescription description2("DummyEventSetupRecordRetriever","",true);
    dummyProv->setDescription(description2);
    provider.add(providerPtr);
    provider.add(finderPtr);
  }
  //checking for conflicts is now delayed until first time EventSetup is requested
  /*EventSetup const& eventSetup = */ provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());

}
void testEventsetup::provenanceTest()
{
  using edm::eventsetup::test::DummyProxyProvider;
  using edm::eventsetup::test::DummyData;
  DummyData kGood{1};
  DummyData kBad{0};

  eventsetup::EventSetupProvider provider(&activityRegistry);
  try {
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","",true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","testLabel",false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
    edm::ESHandle<DummyData> data;
    eventSetup.getData(data);
    CPPUNIT_ASSERT(kGood.value_==data->value_);
    const edm::eventsetup::ComponentDescription* desc = data.description();
    CPPUNIT_ASSERT( desc->label_ == "testLabel");
  } catch (const cms::Exception& iException) {
    std::cout <<"caught "<<iException.explainSelf()<<std::endl;
    throw;
  }
}

void testEventsetup::getDataWithLabelTest()
{
  using edm::eventsetup::test::DummyProxyProvider;
  using edm::eventsetup::test::DummyData;
  DummyData kGood{1};
  DummyData kBad{0};

  eventsetup::EventSetupProvider provider(&activityRegistry);
  try {
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","",true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","testLabel",false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel","blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
    edm::ESHandle<DummyData> data;
    eventSetup.getData("blah",data);
    CPPUNIT_ASSERT(kGood.value_==data->value_);
    const edm::eventsetup::ComponentDescription* desc = data.description();
    CPPUNIT_ASSERT( desc->label_ == "testLabel");
  } catch (const cms::Exception& iException) {
    std::cout <<"caught "<<iException.explainSelf()<<std::endl;
    throw;
  }
}

void testEventsetup::getDataWithESInputTagTest()
{
  using edm::eventsetup::test::DummyProxyProvider;
  using edm::eventsetup::test::DummyData;
  DummyData kGood{1};
  DummyData kBad{0};

  eventsetup::EventSetupProvider provider(&activityRegistry);
  try {
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","testOne",true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","testTwo",false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.addParameter<std::string>("appendToDataLabel","blah");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      dummyProv->setAppendToDataLabel(ps);
      provider.add(dummyProv);
    }
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
    {
      edm::ESHandle<DummyData> data;
      edm::ESInputTag blahTag("","blah");
      eventSetup.getData(blahTag,data);
      CPPUNIT_ASSERT(kGood.value_==data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT( desc->label_ == "testTwo");
    }

    {
      edm::ESHandle<DummyData> data;
      edm::ESInputTag nullTag("","");
      eventSetup.getData(nullTag,data);
      CPPUNIT_ASSERT(kBad.value_==data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT( desc->label_ == "testOne");
    }

    {
      edm::ESHandle<DummyData> data;
      edm::ESInputTag blahTag("testTwo","blah");
      eventSetup.getData(blahTag,data);
      CPPUNIT_ASSERT(kGood.value_==data->value_);
      const edm::eventsetup::ComponentDescription* desc = data.description();
      CPPUNIT_ASSERT( desc->label_ == "testTwo");
    }

    {
      edm::ESHandle<DummyData> data;
      edm::ESInputTag badTag("DoesNotExist","blah");
      CPPUNIT_ASSERT_THROW(eventSetup.getData(badTag, data), cms::Exception);
    }


  } catch (const cms::Exception& iException) {
    std::cout <<"caught "<<iException.explainSelf()<<std::endl;
    throw;
  }
}



void testEventsetup::sourceProducerResolutionTest()
{
  using edm::eventsetup::test::DummyProxyProvider;
  using edm::eventsetup::test::DummyData;
  DummyData kGood{1};
  DummyData kBad{0};

  {
    eventsetup::EventSetupProvider provider(&activityRegistry);
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","",true);
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","",false);
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    //NOTE: use 'invalid' timestamp since the default 'interval of validity'
    //       for a Record is presently an 'invalid' timestamp on both ends.
    //       Since the EventSetup::get<> will only retrieve a Record if its
    //       interval of validity is 'valid' for the present 'instance'
    //       this is a 'hack' to have the 'get' succeed
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
    edm::ESHandle<DummyData> data;
    eventSetup.getData(data);
    CPPUNIT_ASSERT(kGood.value_==data->value_);
  }

  //reverse order
  {
    eventsetup::EventSetupProvider provider(&activityRegistry);
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","",false);
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","",true);
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    //NOTE: use 'invalid' timestamp since the default 'interval of validity'
    //       for a Record is presently an 'invalid' timestamp on both ends.
    //       Since the EventSetup::get<> will only retrieve a Record if its
    //       interval of validity is 'valid' for the present 'instance'
    //       this is a 'hack' to have the 'get' succeed
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
    edm::ESHandle<DummyData> data;
    eventSetup.getData(data);
    CPPUNIT_ASSERT(kGood.value_==data->value_);
  }

}


void testEventsetup::preferTest()
{
  try {
    using edm::eventsetup::test::DummyProxyProvider;
    using edm::eventsetup::test::DummyData;
    DummyData kGood{1};
    DummyData kBad{0};

    {
      using namespace edm::eventsetup;
      EventSetupProvider::PreferredProviderInfo preferInfo;
      EventSetupProvider::RecordToDataMap recordToData;
      //default means use all proxies
      preferInfo[ComponentDescription("DummyProxyProvider","",false)]=recordToData;

      eventsetup::EventSetupProvider provider(&activityRegistry, 0U, &preferInfo);
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider","bad",false);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider","",false);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      //NOTE: use 'invalid' timestamp since the default 'interval of validity'
      //       for a Record is presently an 'invalid' timestamp on both ends.
      //       Since the EventSetup::get<> will only retrieve a Record if its
      //       interval of validity is 'valid' for the present 'instance'
      //       this is a 'hack' to have the 'get' succeed
      EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
      edm::ESHandle<DummyData> data;
      eventSetup.getData(data);
      CPPUNIT_ASSERT(kGood.value_==data->value_);
    }

    //sources
    {
      using namespace edm::eventsetup;
      EventSetupProvider::PreferredProviderInfo preferInfo;
      EventSetupProvider::RecordToDataMap recordToData;
      //default means use all proxies
      preferInfo[ComponentDescription("DummyProxyProvider","",false)]=recordToData;
      eventsetup::EventSetupProvider provider(&activityRegistry, 0U, &preferInfo);
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider","",true);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider","bad",true);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      //NOTE: use 'invalid' timestamp since the default 'interval of validity'
      //       for a Record is presently an 'invalid' timestamp on both ends.
      //       Since the EventSetup::get<> will only retrieve a Record if its
      //       interval of validity is 'valid' for the present 'instance'
      //       this is a 'hack' to have the 'get' succeed
      EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
      edm::ESHandle<DummyData> data;
      eventSetup.getData(data);
      CPPUNIT_ASSERT(kGood.value_==data->value_);
    }

    //specific name
    {
      using namespace edm::eventsetup;
      EventSetupProvider::PreferredProviderInfo preferInfo;
      EventSetupProvider::RecordToDataMap recordToData;
      recordToData.insert(std::make_pair(std::string("DummyRecord"),
                                         std::make_pair(std::string("DummyData"),std::string())));
      preferInfo[ComponentDescription("DummyProxyProvider","",false)]=recordToData;
      eventsetup::EventSetupProvider provider(&activityRegistry, 0U, &preferInfo);
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider","",true);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      {
        edm::eventsetup::ComponentDescription description("DummyProxyProvider","bad",true);
        auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
        dummyProv->setDescription(description);
        provider.add(dummyProv);
      }
      //NOTE: use 'invalid' timestamp since the default 'interval of validity'
      //       for a Record is presently an 'invalid' timestamp on both ends.
      //       Since the EventSetup::get<> will only retrieve a Record if its
      //       interval of validity is 'valid' for the present 'instance'
      //       this is a 'hack' to have the 'get' succeed
      EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
      edm::ESHandle<DummyData> data;
      eventSetup.getData(data);
      CPPUNIT_ASSERT(kGood.value_==data->value_);
    }

  }catch(const cms::Exception& iException) {
    std::cout <<"caught "<<iException.explainSelf()<<std::endl;
    throw;
  }
}

void testEventsetup::introspectionTest()
{
  using edm::eventsetup::test::DummyProxyProvider;
  using edm::eventsetup::test::DummyData;
  DummyData kGood{1};
  DummyData kBad{0};

  eventsetup::EventSetupProvider provider(&activityRegistry);
  try {
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","",true);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test11");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kBad);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    {
      edm::eventsetup::ComponentDescription description("DummyProxyProvider","",false);
      edm::ParameterSet ps;
      ps.addParameter<std::string>("name", "test22");
      ps.registerIt();
      description.pid_ = ps.id();
      auto dummyProv = std::make_shared<DummyProxyProvider>(kGood);
      dummyProv->setDescription(description);
      provider.add(dummyProv);
    }
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());

    std::vector<edm::eventsetup::EventSetupRecordKey> recordKeys;
    eventSetup.fillAvailableRecordKeys(recordKeys);
    CPPUNIT_ASSERT(1==recordKeys.size());
    auto record = eventSetup.find(recordKeys[0]);
    CPPUNIT_ASSERT(record.has_value());
  } catch (const cms::Exception& iException) {
    std::cout <<"caught "<<iException.explainSelf()<<std::endl;
    throw;
  }
}

void testEventsetup::iovExtentionTest()
{
  DummyEventSetupProvider provider(&activityRegistry);
  typedef eventsetup::EventSetupRecordProvider DummyRecordProvider;
  auto dummyRecordProvider = std::make_unique<DummyRecordProvider>(DummyRecord::keyForClass());

  auto finder = std::make_shared<DummyFinder>();
  dummyRecordProvider->addFinder(finder);

  provider.insert(std::move(dummyRecordProvider));

  const Timestamp time_2(2);
  finder->setInterval(ValidityInterval(IOVSyncValue{time_2}, IOVSyncValue{Timestamp{3}}));
  {
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue{time_2});
    CPPUNIT_ASSERT(2==eventSetup.get<DummyRecord>().cacheIdentifier());
  }
  {
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue{Timestamp{3}});
    eventSetup.get<DummyRecord>();
    CPPUNIT_ASSERT(2==eventSetup.get<DummyRecord>().cacheIdentifier());
  }
  //extending the IOV should not cause the cache to be reset
  finder->setInterval(ValidityInterval(IOVSyncValue{time_2}, IOVSyncValue{Timestamp{4}}));
  {
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue{Timestamp{4}});
    CPPUNIT_ASSERT(2==eventSetup.get<DummyRecord>().cacheIdentifier());
  }

  //this is a new IOV so should get cache reset
  finder->setInterval(ValidityInterval(IOVSyncValue{Timestamp{5}}, IOVSyncValue{Timestamp{6}}));
  {
    EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue{Timestamp{5}});
    CPPUNIT_ASSERT(3==eventSetup.get<DummyRecord>().cacheIdentifier());
  }

}
