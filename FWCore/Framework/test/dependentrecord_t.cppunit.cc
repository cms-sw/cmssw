/*
 *  dependentrecord_t.cpp
 *  EDMProto
 *
 *  Created by Chris Jones on 4/29/05.
 *  Changed by Viji Sundararajan on 29-Jun-2005
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/DepRecord.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordProviderFactoryManager.h"
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/NoRecordException.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace edm::eventsetup;

class testdependentrecord: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testdependentrecord);

CPPUNIT_TEST(dependentConstructorTest);
CPPUNIT_TEST(dependentFinder1Test);
CPPUNIT_TEST(dependentFinder2Test);
CPPUNIT_TEST(dependentSetproviderTest);
CPPUNIT_TEST_EXCEPTION(getTest,edm::eventsetup::NoRecordException<DepRecord>);

CPPUNIT_TEST_SUITE_END();
public:

  void setUp(){}
  void tearDown(){}

  void dependentConstructorTest();
  void dependentFinder1Test();
  void dependentFinder2Test();
  void dependentSetproviderTest();
  void getTest();

}; //Cppunit class declaration over

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testdependentrecord);

class DummyProxyProvider : public edm::eventsetup::DataProxyProvider {
public:
   DummyProxyProvider() {
      usingRecord<DummyRecord>();
   }
   void newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType,
                     const edm::ValidityInterval& iInterval) {
      //do nothing
   }
protected:
   void registerProxies(const edm::eventsetup::EventSetupRecordKey&, KeyedProxies& iHolder) {
   }
   
};

#include "FWCore/Framework/interface/DataProxyProvider.h"

class DepRecordProxyProvider : public edm::eventsetup::DataProxyProvider {
public:
   DepRecordProxyProvider() {
      usingRecord<DepRecord>();
   }
   void newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType,
                     const edm::ValidityInterval& iInterval) {
      //do nothing
   }
protected:
   void registerProxies(const edm::eventsetup::EventSetupRecordKey&, KeyedProxies& iHolder) {
   }
   
};

using namespace edm::eventsetup;
void testdependentrecord::dependentConstructorTest()
{
   std::auto_ptr<EventSetupRecordProvider> depProvider =
   EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(DepRecord::keyForClass());
   
   CPPUNIT_ASSERT(1 == depProvider->dependentRecords().size());
   CPPUNIT_ASSERT(*(depProvider->dependentRecords().begin()) == DummyRecord::keyForClass());
}


void testdependentrecord::dependentFinder1Test()
{
   boost::shared_ptr<EventSetupRecordProvider> dummyProvider(
                                                          EventSetupRecordProviderFactoryManager::instance()
                                                              .makeRecordProvider(DummyRecord::keyForClass()).release());
   
   const edm::ValidityInterval definedInterval(1,3);
   boost::shared_ptr<DummyFinder> dummyFinder(new DummyFinder);
   dummyFinder->setInterval(definedInterval);
   dummyProvider->addFinder(dummyFinder);
   
   const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
   DependentRecordIntervalFinder finder(depRecordKey);
   finder.addProviderWeAreDependentOn(dummyProvider);
   
   CPPUNIT_ASSERT(definedInterval == finder.findIntervalFor(depRecordKey, edm::Timestamp(2))); 

   dummyFinder->setInterval(edm::ValidityInterval::invalidInterval());
   CPPUNIT_ASSERT(edm::ValidityInterval::invalidInterval() == finder.findIntervalFor(depRecordKey, edm::Timestamp(4)));
   
   const edm::ValidityInterval unknownedEndInterval(5,edm::Timestamp::invalidTimestamp());
   dummyFinder->setInterval(unknownedEndInterval);

   CPPUNIT_ASSERT(unknownedEndInterval == finder.findIntervalFor(depRecordKey, edm::Timestamp(5)));

}

void testdependentrecord::dependentFinder2Test()
{
   boost::shared_ptr<EventSetupRecordProvider> dummyProvider1(EventSetupRecordProviderFactoryManager::instance()
                                                              .makeRecordProvider(DummyRecord::keyForClass()).release());
   
   const edm::ValidityInterval definedInterval1(1,5);
   dummyProvider1->setValidityInterval(definedInterval1);
   
   boost::shared_ptr<EventSetupRecordProvider> dummyProvider2(EventSetupRecordProviderFactoryManager::instance()
                                                              .makeRecordProvider(DummyRecord::keyForClass()).release());
   
   const edm::ValidityInterval definedInterval2(2,6);
   dummyProvider2->setValidityInterval(definedInterval2);

   const edm::ValidityInterval overlapInterval(std::max(definedInterval1.first(), definedInterval2.first()),
                                                std::min(definedInterval1.last(), definedInterval2.last()));
   
   const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
                                                     
   DependentRecordIntervalFinder finder(depRecordKey);
   finder.addProviderWeAreDependentOn(dummyProvider1);
   finder.addProviderWeAreDependentOn(dummyProvider2);
   
   CPPUNIT_ASSERT(overlapInterval == finder.findIntervalFor(depRecordKey, edm::Timestamp(4)));
}


void testdependentrecord::dependentSetproviderTest()
{
   std::auto_ptr<EventSetupRecordProvider> depProvider =
   EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(DepRecord::keyForClass());
   
   boost::shared_ptr<EventSetupRecordProvider> dummyProvider(
       EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(DummyRecord::keyForClass()).release());

   boost::shared_ptr<DummyFinder> dummyFinder(new DummyFinder);
   dummyFinder->setInterval(edm::ValidityInterval(1, 3));
   dummyProvider->addFinder(dummyFinder);
   
   CPPUNIT_ASSERT(*(depProvider->dependentRecords().begin()) == dummyProvider->key());
   
   std::vector< boost::shared_ptr<EventSetupRecordProvider> > providers;
   providers.push_back(dummyProvider);
   depProvider->setDependentProviders(providers);
}

void testdependentrecord::getTest()
{
   edm::eventsetup::EventSetupProvider provider;
   boost::shared_ptr<edm::eventsetup::DataProxyProvider> dummyProv(new DummyProxyProvider());
   provider.add(dummyProv);

   boost::shared_ptr<DummyFinder> dummyFinder(new DummyFinder);
   dummyFinder->setInterval(edm::ValidityInterval(1, 3));
   provider.add(boost::shared_ptr<edm::eventsetup::EventSetupRecordIntervalFinder>(dummyFinder));
   
   boost::shared_ptr<edm::eventsetup::DataProxyProvider> depProv(new DepRecordProxyProvider());
   provider.add(depProv);
   {
      const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::Timestamp(1));
      const DepRecord& depRecord = eventSetup.get<DepRecord>();

      depRecord.getRecord<DummyRecord>();
   }
   {
      const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::Timestamp(4));
      eventSetup.get<DepRecord>();
   }
}

