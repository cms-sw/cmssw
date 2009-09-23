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
#include "FWCore/Framework/test/Dummy2Record.h"
#include "FWCore/Framework/test/DepRecord.h"
#include "FWCore/Framework/test/DepOn2Record.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordProviderFactoryManager.h"
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/interface/print_eventsetup_record_dependencies.h"

#include <cppunit/extensions/HelperMacros.h>
#include <string.h>


using namespace edm::eventsetup;

class testdependentrecord: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testdependentrecord);

CPPUNIT_TEST(dependentConstructorTest);
CPPUNIT_TEST(dependentFinder1Test);
CPPUNIT_TEST(dependentFinder2Test);
CPPUNIT_TEST(dependentSetproviderTest);
CPPUNIT_TEST(getTest);
CPPUNIT_TEST(oneOfTwoRecordTest);
CPPUNIT_TEST(resetTest);
CPPUNIT_TEST(alternateFinderTest);

CPPUNIT_TEST_SUITE_END();
public:

  void setUp(){}
  void tearDown(){}

  void dependentConstructorTest();
  void dependentFinder1Test();
  void dependentFinder2Test();
  void dependentSetproviderTest();
  void getTest();
  void oneOfTwoRecordTest();
  void resetTest();
  void alternateFinderTest();
  
}; //Cppunit class declaration over

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testdependentrecord);

namespace {
class DummyProxyProvider : public edm::eventsetup::DataProxyProvider {
public:
   DummyProxyProvider() {
      usingRecord<DummyRecord>();
   }
   void newInterval(const edm::eventsetup::EventSetupRecordKey& /*iRecordType*/,
                     const edm::ValidityInterval& /*iInterval*/) {
      //do nothing
   }
protected:
   void registerProxies(const edm::eventsetup::EventSetupRecordKey&, KeyedProxies& /*iHolder*/) {
   }
   
};

class DepRecordProxyProvider : public edm::eventsetup::DataProxyProvider {
public:
   DepRecordProxyProvider() {
      usingRecord<DepRecord>();
   }
   void newInterval(const edm::eventsetup::EventSetupRecordKey& /*iRecordType*/,
                     const edm::ValidityInterval& /*iInterval*/) {
      //do nothing
   }
protected:
   void registerProxies(const edm::eventsetup::EventSetupRecordKey&, KeyedProxies& /*iHolder*/) {
   }
   
};


class DepOn2RecordProxyProvider : public edm::eventsetup::DataProxyProvider {
public:
  DepOn2RecordProxyProvider() {
    usingRecord<DepOn2Record>();
  }
  void newInterval(const edm::eventsetup::EventSetupRecordKey& /*iRecordType*/,
  const edm::ValidityInterval& /*iInterval*/) {
    //do nothing
  }
protected:
void registerProxies(const edm::eventsetup::EventSetupRecordKey&, KeyedProxies& /*iHolder*/) {
}

};

class DepRecordFinder : public edm::EventSetupRecordIntervalFinder {
public:
  DepRecordFinder() :edm::EventSetupRecordIntervalFinder(), interval_() {
    this->findingRecord<DepRecord>();
  }
  
  void setInterval(const edm::ValidityInterval& iInterval) {
    interval_ = iInterval;
  }
protected:
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                              const edm::IOVSyncValue& iTime, 
                              edm::ValidityInterval& iInterval) {
    if(interval_.validFor(iTime)) {
      iInterval = interval_;
    } else {
      if(interval_.last() == edm::IOVSyncValue::invalidIOVSyncValue() &&
         interval_.first() != edm::IOVSyncValue::invalidIOVSyncValue() &&
         interval_.first() <= iTime) {
        iInterval = interval_;
      }else {
        iInterval = edm::ValidityInterval();
      }
    }
  }
private:
  edm::ValidityInterval interval_;   
};
}

using namespace edm::eventsetup;
void testdependentrecord::dependentConstructorTest()
{
   std::auto_ptr<EventSetupRecordProvider> depProvider =
   EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(DepRecord::keyForClass());
   
   CPPUNIT_ASSERT(1 == depProvider->dependentRecords().size());
   CPPUNIT_ASSERT(*(depProvider->dependentRecords().begin()) == DummyRecord::keyForClass());
   
   edm::print_eventsetup_record_dependencies<DepRecord>(std::cout);
}


void testdependentrecord::dependentFinder1Test()
{
   boost::shared_ptr<EventSetupRecordProvider> dummyProvider(
                                                          EventSetupRecordProviderFactoryManager::instance()
                                                              .makeRecordProvider(DummyRecord::keyForClass()).release());
   const edm::EventID eID_1(1, 1, 1);
   const edm::IOVSyncValue sync_1(eID_1);
   const edm::EventID eID_3(1, 1, 3);
   const edm::ValidityInterval definedInterval(sync_1, 
                                                edm::IOVSyncValue(eID_3));
   boost::shared_ptr<DummyFinder> dummyFinder(new DummyFinder);
   dummyFinder->setInterval(definedInterval);
   dummyProvider->addFinder(dummyFinder);
   
   const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
   DependentRecordIntervalFinder finder(depRecordKey);
   finder.addProviderWeAreDependentOn(dummyProvider);
   
   CPPUNIT_ASSERT(definedInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 2)))); 

   dummyFinder->setInterval(edm::ValidityInterval::invalidInterval());
   CPPUNIT_ASSERT(edm::ValidityInterval::invalidInterval() == finder.findIntervalFor(depRecordKey, 
                                                                                     edm::IOVSyncValue(edm::EventID(1, 1, 4))));
   
   const edm::EventID eID_5(1, 1, 5);
   const edm::IOVSyncValue sync_5(eID_5);
   const edm::ValidityInterval unknownedEndInterval(sync_5 ,
                                                     edm::IOVSyncValue::invalidIOVSyncValue());
   dummyFinder->setInterval(unknownedEndInterval);

   CPPUNIT_ASSERT(unknownedEndInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5))));

}

void testdependentrecord::dependentFinder2Test()
{
   boost::shared_ptr<EventSetupRecordProvider> dummyProvider1(EventSetupRecordProviderFactoryManager::instance()
                                                              .makeRecordProvider(DummyRecord::keyForClass()).release());
   
   const edm::EventID eID_1(1, 1, 1);
   const edm::IOVSyncValue sync_1(eID_1);
   const edm::ValidityInterval definedInterval1(sync_1, 
                                                 edm::IOVSyncValue(edm::EventID(1, 1, 5)));
   dummyProvider1->setValidityInterval(definedInterval1);
   
   boost::shared_ptr<EventSetupRecordProvider> dummyProvider2(EventSetupRecordProviderFactoryManager::instance()
                                                              .makeRecordProvider(DummyRecord::keyForClass()).release());
   
   const edm::EventID eID_2(1, 1, 2);
   const edm::IOVSyncValue sync_2(eID_2);
   const edm::ValidityInterval definedInterval2(sync_2, 
                                                 edm::IOVSyncValue(edm::EventID(1, 1, 6)));
   dummyProvider2->setValidityInterval(definedInterval2);

   const edm::ValidityInterval overlapInterval(std::max(definedInterval1.first(), definedInterval2.first()),
                                                std::min(definedInterval1.last(), definedInterval2.last()));
   
   const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
                                                     
   DependentRecordIntervalFinder finder(depRecordKey);
   finder.addProviderWeAreDependentOn(dummyProvider1);
   finder.addProviderWeAreDependentOn(dummyProvider2);
   
   CPPUNIT_ASSERT(overlapInterval == finder.findIntervalFor(depRecordKey, 
                                                            edm::IOVSyncValue(edm::EventID(1, 1, 4))));
}


void testdependentrecord::dependentSetproviderTest()
{
   std::auto_ptr<EventSetupRecordProvider> depProvider =
   EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(DepRecord::keyForClass());
   
   boost::shared_ptr<EventSetupRecordProvider> dummyProvider(
       EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(DummyRecord::keyForClass()).release());

   boost::shared_ptr<DummyFinder> dummyFinder(new DummyFinder);
   dummyFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)),
                                                   edm::IOVSyncValue(edm::EventID(1, 1, 3))));
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
   dummyFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), 
                                                   edm::IOVSyncValue(edm::EventID(1, 1, 3))));
   provider.add(boost::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));
   
   boost::shared_ptr<edm::eventsetup::DataProxyProvider> depProv(new DepRecordProxyProvider());
   provider.add(depProv);
   {
      const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
      const DepRecord& depRecord = eventSetup.get<DepRecord>();

      depRecord.getRecord<DummyRecord>();
   }
   {
      const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 4)));
      CPPUNIT_ASSERT_THROW(eventSetup.get<DepRecord>(),edm::eventsetup::NoRecordException<DepRecord>);
   }
   
}

void testdependentrecord::oneOfTwoRecordTest()
{
  edm::eventsetup::EventSetupProvider provider;
  boost::shared_ptr<edm::eventsetup::DataProxyProvider> dummyProv(new DummyProxyProvider());
  provider.add(dummyProv);
  
  boost::shared_ptr<DummyFinder> dummyFinder(new DummyFinder);
  dummyFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), 
                                                 edm::IOVSyncValue(edm::EventID(1, 1, 3))));
  provider.add(boost::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));
  
  boost::shared_ptr<edm::eventsetup::DataProxyProvider> depProv(new DepOn2RecordProxyProvider());
  provider.add(depProv);
  {
    const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    const DepOn2Record& depRecord = eventSetup.get<DepOn2Record>();
    
    depRecord.getRecord<DummyRecord>();
    CPPUNIT_ASSERT_THROW(depRecord.getRecord<Dummy2Record>(),edm::eventsetup::NoRecordException<Dummy2Record>);

    try {
      depRecord.getRecord<Dummy2Record>();
    } catch(edm::eventsetup::NoRecordException<Dummy2Record>& e) {
       //make sure that the record name appears in the error message.
       CPPUNIT_ASSERT(0!=strstr(e.what(), "DepOn2Record"));
       CPPUNIT_ASSERT(0!=strstr(e.what(), "Dummy2Record"));
       //	std::cout<<e.what()<<std::endl;
    }
  }
}
void testdependentrecord::resetTest()
{
  edm::eventsetup::EventSetupProvider provider;
  boost::shared_ptr<edm::eventsetup::DataProxyProvider> dummyProv(new DummyProxyProvider());
  provider.add(dummyProv);
  
  boost::shared_ptr<DummyFinder> dummyFinder(new DummyFinder);
  dummyFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)), 
                                                 edm::IOVSyncValue(edm::EventID(1, 1, 3))));
  provider.add(boost::shared_ptr<edm::EventSetupRecordIntervalFinder>(dummyFinder));
  
  boost::shared_ptr<edm::eventsetup::DataProxyProvider> depProv(new DepRecordProxyProvider());
  provider.add(depProv);
  {
    const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(edm::EventID(1, 1, 1)));
    const DepRecord& depRecord = eventSetup.get<DepRecord>();
    unsigned long long depCacheID = depRecord.cacheIdentifier();
    const DummyRecord& dummyRecord = depRecord.getRecord<DummyRecord>();
    unsigned long long dummyCacheID = dummyRecord.cacheIdentifier();
    
    provider.resetRecordPlusDependentRecords(dummyRecord.key());
    CPPUNIT_ASSERT(dummyCacheID != dummyRecord.cacheIdentifier());
    CPPUNIT_ASSERT(depCacheID != depRecord.cacheIdentifier());
  }
}
void testdependentrecord::alternateFinderTest()
{
  boost::shared_ptr<EventSetupRecordProvider> dummyProvider(
                                                            EventSetupRecordProviderFactoryManager::instance()
                                                            .makeRecordProvider(DummyRecord::keyForClass()).release());
  const edm::EventID eID_1(1, 1, 1);
  const edm::IOVSyncValue sync_1(eID_1);
  const edm::EventID eID_3(1, 1, 3);
  const edm::IOVSyncValue sync_3(eID_3);
  const edm::EventID eID_4(1, 1, 4);
  const edm::ValidityInterval definedInterval(sync_1, 
                                              edm::IOVSyncValue(eID_4));
  boost::shared_ptr<DummyFinder> dummyFinder(new DummyFinder);
  dummyFinder->setInterval(definedInterval);
  dummyProvider->addFinder(dummyFinder);
  
  boost::shared_ptr<DepRecordFinder> depFinder(new DepRecordFinder);
  const edm::EventID eID_2(1, 1, 2);
  const edm::IOVSyncValue sync_2(eID_2);
  const edm::ValidityInterval depInterval(sync_1, 
                                          sync_2);
  depFinder->setInterval(depInterval);
  
  const EventSetupRecordKey depRecordKey = DepRecord::keyForClass();
  DependentRecordIntervalFinder finder(depRecordKey);
  finder.setAlternateFinder(depFinder);
  finder.addProviderWeAreDependentOn(dummyProvider);
  
  CPPUNIT_ASSERT(depInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 1)))); 
  
  const edm::ValidityInterval dep2Interval(sync_3, 
                                           edm::IOVSyncValue(eID_4));
  depFinder->setInterval(dep2Interval);
  const edm::ValidityInterval tempIOV =finder.findIntervalFor(depRecordKey, sync_3);
  //std::cout <<  tempIOV.first().eventID()<<" to "<<tempIOV.last().eventID() <<std::endl;
  CPPUNIT_ASSERT(dep2Interval == finder.findIntervalFor(depRecordKey, sync_3)); 
  
  dummyFinder->setInterval(edm::ValidityInterval::invalidInterval());
  depFinder->setInterval(edm::ValidityInterval::invalidInterval());
  CPPUNIT_ASSERT(edm::ValidityInterval::invalidInterval() == finder.findIntervalFor(depRecordKey, 
                                                                                    edm::IOVSyncValue(edm::EventID(1, 1, 5))));
  
  const edm::EventID eID_6(1, 1, 6);
  const edm::IOVSyncValue sync_6(eID_6);
  const edm::ValidityInterval unknownedEndInterval(sync_6 ,
                                                   edm::IOVSyncValue::invalidIOVSyncValue());
  dummyFinder->setInterval(unknownedEndInterval);
  
  const edm::EventID eID_7(1, 1, 7);
  const edm::IOVSyncValue sync_7(eID_7);
  const edm::ValidityInterval iov6_7(sync_6,sync_7);
  depFinder->setInterval(iov6_7);
  
  CPPUNIT_ASSERT(unknownedEndInterval == finder.findIntervalFor(depRecordKey, sync_6));
  
  //see if dependent record can override the finder
  dummyFinder->setInterval(depInterval);
  depFinder->setInterval(definedInterval);
  CPPUNIT_ASSERT(depInterval == finder.findIntervalFor(depRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 1)))); 

  dummyFinder->setInterval(dep2Interval);
  CPPUNIT_ASSERT(dep2Interval == finder.findIntervalFor(depRecordKey, sync_3)); 
}
