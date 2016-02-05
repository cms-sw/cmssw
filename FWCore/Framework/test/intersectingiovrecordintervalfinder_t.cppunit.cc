// -*- C++ -*-
//
// Package:     Framework
// Class  :     intersectingiovrecordintervalfinder_t_cppunit
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Aug 19 14:14:42 EDT 2008
//

// system include files

// user include files
#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/DummyFinder.h"

#include "cppunit/extensions/HelperMacros.h"
using namespace edm::eventsetup;

class testintersectingiovrecordintervalfinder: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testintersectingiovrecordintervalfinder);
   
   CPPUNIT_TEST(constructorTest);
   CPPUNIT_TEST(intersectionTest);
   
   CPPUNIT_TEST_SUITE_END();
public:
   
   void setUp(){}
   void tearDown(){}
   
   void constructorTest();
   void intersectionTest();
   
}; //Cppunit class declaration over

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testintersectingiovrecordintervalfinder);
namespace  {
   class DepRecordFinder : public edm::EventSetupRecordIntervalFinder {
   public:
      DepRecordFinder() :edm::EventSetupRecordIntervalFinder(), interval_() {
         this->findingRecord<DummyRecord>();
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

void 
testintersectingiovrecordintervalfinder::constructorTest()
{
   IntersectingIOVRecordIntervalFinder finder(DummyRecord::keyForClass());
   CPPUNIT_ASSERT(finder.findingForRecords().size() == 1);
   std::set<EventSetupRecordKey> s = finder.findingForRecords();
   CPPUNIT_ASSERT(s.find(DummyRecord::keyForClass()) != s.end());
}

void 
testintersectingiovrecordintervalfinder::intersectionTest()
{
   const EventSetupRecordKey dummyRecordKey = DummyRecord::keyForClass();

   IntersectingIOVRecordIntervalFinder intFinder(dummyRecordKey);
   std::vector<edm::propagate_const<std::shared_ptr<edm::EventSetupRecordIntervalFinder>>> finders;
   std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
   {
      const edm::EventID eID_1(1, 1, 1);
      const edm::IOVSyncValue sync_1(eID_1);
      const edm::EventID eID_3(1, 1, 3);
      const edm::ValidityInterval definedInterval(sync_1, 
                                                  edm::IOVSyncValue(eID_3));
      finders.push_back(dummyFinder);
      dummyFinder->setInterval(definedInterval);
      intFinder.swapFinders(finders);
      
      CPPUNIT_ASSERT(definedInterval == intFinder.findIntervalFor(dummyRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 2)))); 
   }

   {
      const edm::EventID eID_5(1, 1, 5);
      const edm::IOVSyncValue sync_5(eID_5);
      const edm::ValidityInterval unknownedEndInterval(sync_5 ,
                                                       edm::IOVSyncValue::invalidIOVSyncValue());
      dummyFinder->setInterval(unknownedEndInterval);
   
      CPPUNIT_ASSERT(unknownedEndInterval == intFinder.findIntervalFor(dummyRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5))));
   }
   
   {
      finders.clear();
      
      const edm::EventID eID_1(1, 1, 1);
      const edm::IOVSyncValue sync_1(eID_1);
      const edm::EventID eID_3(1, 1, 3);
      const edm::IOVSyncValue sync_3(eID_3);
      const edm::EventID eID_4(1, 1, 4);
      const edm::IOVSyncValue sync_4(eID_4);
      const edm::EventID eID_5(1, 1, 5);
      const edm::IOVSyncValue sync_5(eID_5);
      const edm::ValidityInterval definedInterval(sync_1, 
                                                  sync_4);
      dummyFinder->setInterval(definedInterval);
      finders.push_back(dummyFinder);

      std::shared_ptr<DummyFinder> dummyFinder2 = std::make_shared<DummyFinder>();
      dummyFinder2->setInterval(edm::ValidityInterval(sync_3, sync_5));
      finders.push_back(dummyFinder2);
      IntersectingIOVRecordIntervalFinder intFinder(dummyRecordKey);
      intFinder.swapFinders(finders);

      CPPUNIT_ASSERT(edm::ValidityInterval(sync_3,sync_4) ==
                     intFinder.findIntervalFor(dummyRecordKey, sync_3));
   }
   
   {
      finders.clear();
      const edm::EventID eID_1(1, 1, 1);
      const edm::IOVSyncValue sync_1(eID_1);
      const edm::EventID eID_3(1, 1, 3);
      const edm::IOVSyncValue sync_3(eID_3);
      const edm::EventID eID_4(1, 1, 4);
      const edm::IOVSyncValue sync_4(eID_4);
      const edm::EventID eID_5(1, 1, 5);
      const edm::IOVSyncValue sync_5(eID_5);
      const edm::ValidityInterval definedInterval(sync_1, 
                                                  sync_4);
      dummyFinder->setInterval(definedInterval);
      finders.push_back(dummyFinder);
      
      std::shared_ptr<DummyFinder> dummyFinder2 = std::make_shared<DummyFinder>();
      dummyFinder2->setInterval(edm::ValidityInterval::invalidInterval());
      finders.push_back(dummyFinder2);
      IntersectingIOVRecordIntervalFinder intFinder(dummyRecordKey);
      intFinder.swapFinders(finders);
      
      CPPUNIT_ASSERT(edm::ValidityInterval::invalidInterval() ==
                     dummyFinder2->findIntervalFor(dummyRecordKey,sync_3));
      
      CPPUNIT_ASSERT(edm::ValidityInterval(sync_1,edm::IOVSyncValue::invalidIOVSyncValue()) ==
                     intFinder.findIntervalFor(dummyRecordKey, sync_3));
   }
   
   {
      finders.clear();
      const edm::EventID eID_1(1, 1, 1);
      const edm::IOVSyncValue sync_1(eID_1);
      const edm::EventID eID_3(1, 1, 3);
      const edm::IOVSyncValue sync_3(eID_3);
      const edm::EventID eID_4(1, 1, 4);
      const edm::IOVSyncValue sync_4(eID_4);
      const edm::ValidityInterval definedInterval(sync_1, 
                                                  sync_4);
      dummyFinder->setInterval(definedInterval);
      finders.push_back(dummyFinder);
      
      std::shared_ptr<DummyFinder> dummyFinder2 = std::make_shared<DummyFinder>();
      dummyFinder2->setInterval(edm::ValidityInterval(sync_3, edm::IOVSyncValue::invalidIOVSyncValue()));
      finders.push_back(dummyFinder2);
      IntersectingIOVRecordIntervalFinder intFinder(dummyRecordKey);
      intFinder.swapFinders(finders);
      
      CPPUNIT_ASSERT(edm::ValidityInterval(sync_3,edm::IOVSyncValue::invalidIOVSyncValue()) ==
                     dummyFinder2->findIntervalFor(dummyRecordKey,sync_3));
      
      CPPUNIT_ASSERT(edm::ValidityInterval(sync_3,edm::IOVSyncValue::invalidIOVSyncValue()) ==
                     intFinder.findIntervalFor(dummyRecordKey, sync_3));
   }

   {
      //reverse order so invalid ending is first in list
      finders.clear();
      const edm::EventID eID_1(1, 1, 1);
      const edm::IOVSyncValue sync_1(eID_1);
      const edm::EventID eID_3(1, 1, 3);
      const edm::IOVSyncValue sync_3(eID_3);
      const edm::EventID eID_4(1, 1, 4);
      const edm::IOVSyncValue sync_4(eID_4);
      const edm::ValidityInterval definedInterval(sync_1, 
                                                  sync_4);
      
      std::shared_ptr<DummyFinder> dummyFinder2 = std::make_shared<DummyFinder>();
      dummyFinder2->setInterval(edm::ValidityInterval(sync_3, edm::IOVSyncValue::invalidIOVSyncValue()));
      finders.push_back(dummyFinder2);

      dummyFinder->setInterval(definedInterval);
      finders.push_back(dummyFinder);

      IntersectingIOVRecordIntervalFinder intFinder(dummyRecordKey);
      intFinder.swapFinders(finders);
      
      CPPUNIT_ASSERT(edm::ValidityInterval(sync_3,edm::IOVSyncValue::invalidIOVSyncValue()) ==
                     dummyFinder2->findIntervalFor(dummyRecordKey,sync_3));
      
      CPPUNIT_ASSERT(edm::ValidityInterval(sync_3,edm::IOVSyncValue::invalidIOVSyncValue()) ==
                     intFinder.findIntervalFor(dummyRecordKey, sync_3));
   }
   
}
