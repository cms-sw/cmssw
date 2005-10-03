/*
 *  eventsetup_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/24/05.
 *  Changed by Viji Sundararajan on 24-Jun-05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "FWCore/Framework/interface/eventSetupGetImplementation.icc"

#include "FWCore/Framework/test/DummyRecord.h"
//class DummyRecord : public edm::eventsetup::EventSetupRecordImplementation<DummyRecord> {};

#include "FWCore/Framework/interface/HCMethods.icc"
//#include "FWCore/Framework/interface/HCTypeTag.icc"
#include "FWCore/Framework/interface/HCTypeTagTemplate.h"
/*
template<>
const char*
edm::eventsetup::heterocontainer::HCTypeTagTemplate<DummyRecord, edm::eventsetup::EventSetupRecordKey>::className() {
   return "DummyRecord";
}
*/
using namespace edm;

class testEventsetup: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testEventsetup);

CPPUNIT_TEST(constructTest);
CPPUNIT_TEST(getTest);
CPPUNIT_TEST_EXCEPTION(getExcTest,edm::eventsetup::NoRecordException<DummyRecord>);
CPPUNIT_TEST(recordProviderTest);
CPPUNIT_TEST_EXCEPTION(recordValidityTest,edm::eventsetup::NoRecordException<DummyRecord>);
CPPUNIT_TEST_EXCEPTION(recordValidityExcTest,edm::eventsetup::NoRecordException<DummyRecord>);
CPPUNIT_TEST(proxyProviderTest);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void constructTest();
  void getTest();
  void getExcTest();
  void recordProviderTest();
  void recordValidityTest();
  void recordValidityExcTest();
  void proxyProviderTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventsetup);


void testEventsetup::constructTest()
{
   eventsetup::EventSetupProvider provider;
   const Timestamp time(1);
   const IOVSyncValue timestamp(time);
   EventSetup const& eventSetup = provider.eventSetupForInstance(timestamp);
   CPPUNIT_ASSERT(&eventSetup != 0);
   CPPUNIT_ASSERT(eventSetup.iovSyncValue() == timestamp);
}

void testEventsetup::getTest()
{
   eventsetup::EventSetupProvider provider;
   EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
   CPPUNIT_ASSERT(&eventSetup != 0);
   //eventSetup.get<DummyRecord>();
   //BOOST_CHECK_THROW(eventSetup.get<DummyRecord>(), edm::eventsetup::NoRecordException<DummyRecord>);
   
   DummyRecord dummyRecord;
   provider.addRecordToEventSetup(dummyRecord);
   const DummyRecord& gottenRecord = eventSetup.get<DummyRecord>();
   CPPUNIT_ASSERT(0 != &gottenRecord);
   CPPUNIT_ASSERT(&dummyRecord == &gottenRecord);
}

void testEventsetup::getExcTest()
{
   eventsetup::EventSetupProvider provider;
   EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
   CPPUNIT_ASSERT(&eventSetup != 0);
   eventSetup.get<DummyRecord>();
   //BOOST_CHECK_THROW(eventSetup.get<DummyRecord>(), edm::eventsetup::NoRecordException<DummyRecord>);
}

#include "FWCore/Framework/interface/EventSetupRecordProviderTemplate.h"

class DummyEventSetupProvider : public edm::eventsetup::EventSetupProvider {
public:
   template<class T>
   void insert(std::auto_ptr<T> iRecord) {
      edm::eventsetup::EventSetupProvider::insert(iRecord);
   }
};

void testEventsetup::recordProviderTest()
{
   DummyEventSetupProvider provider;
   typedef eventsetup::EventSetupRecordProviderTemplate<DummyRecord> DummyRecordProvider;
   std::auto_ptr<DummyRecordProvider > dummyRecordProvider(new DummyRecordProvider());
   
   provider.insert(dummyRecordProvider);
   
   //NOTE: use 'invalid' timestamp since the default 'interval of validity'
   //       for a Record is presently an 'invalid' timestamp on both ends.
   //       Since the EventSetup::get<> will only retrieve a Record if its
   //       interval of validity is 'valid' for the present 'instance'
   //       this is a 'hack' to have the 'get' succeed
   EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
   const DummyRecord& gottenRecord = eventSetup.get<DummyRecord>();
   CPPUNIT_ASSERT(0 != &gottenRecord);
}

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

class DummyFinder : public EventSetupRecordIntervalFinder {
public:
   DummyFinder() {
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
   DummyEventSetupProvider provider;
   typedef eventsetup::EventSetupRecordProviderTemplate<DummyRecord> DummyRecordProvider;
   std::auto_ptr<DummyRecordProvider > dummyRecordProvider(new DummyRecordProvider());

   boost::shared_ptr<DummyFinder> finder(new DummyFinder);
   dummyRecordProvider->addFinder(finder);
   
   provider.insert(dummyRecordProvider);
   
   {
      Timestamp time_1(1);
      /*EventSetup const& eventSetup = */ provider.eventSetupForInstance(IOVSyncValue(time_1));
   // BOOST_CHECK_THROW(eventSetup.get<DummyRecord>(), edm::eventsetup::NoRecordException<DummyRecord>);
   //eventSetup.get<DummyRecord>();
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
   //   BOOST_CHECK_THROW(eventSetup.get<DummyRecord>(), edm::eventsetup::NoRecordException<DummyRecord>);
   eventSetup.get<DummyRecord>();
   }
   
   
}

void testEventsetup::recordValidityExcTest()
{
   DummyEventSetupProvider provider;
   typedef eventsetup::EventSetupRecordProviderTemplate<DummyRecord> DummyRecordProvider;
   std::auto_ptr<DummyRecordProvider > dummyRecordProvider(new DummyRecordProvider());

   boost::shared_ptr<DummyFinder> finder(new DummyFinder);
   dummyRecordProvider->addFinder(finder);
   
   provider.insert(dummyRecordProvider);
   
   {
      EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue(Timestamp(1)));
   // BOOST_CHECK_THROW(eventSetup.get<DummyRecord>(), edm::eventsetup::NoRecordException<DummyRecord>);
   eventSetup.get<DummyRecord>();
   }

}
#include "FWCore/Framework/interface/DataProxyProvider.h"

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

#include "FWCore/Framework/interface/EventSetupRecordProviderFactoryTemplate.h"
//create an instance of the factory
static eventsetup::EventSetupRecordProviderFactoryTemplate<DummyRecord> s_factory;

void testEventsetup::proxyProviderTest()
{
   eventsetup::EventSetupProvider provider;
   boost::shared_ptr<eventsetup::DataProxyProvider> dummyProv(new DummyProxyProvider());
   provider.add(dummyProv);
   
   EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue::invalidIOVSyncValue());
   const DummyRecord& gottenRecord = eventSetup.get<DummyRecord>();
   CPPUNIT_ASSERT(0 != &gottenRecord);
}
