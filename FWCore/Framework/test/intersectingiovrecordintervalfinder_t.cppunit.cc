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

#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/DummyFinder.h"

#include "cppunit/extensions/HelperMacros.h"
using namespace edm::eventsetup;

class testintersectingiovrecordintervalfinder : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testintersectingiovrecordintervalfinder);

  CPPUNIT_TEST(constructorTest);
  CPPUNIT_TEST(intersectionTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void constructorTest();
  void intersectionTest();

};  //Cppunit class declaration over

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testintersectingiovrecordintervalfinder);

void testintersectingiovrecordintervalfinder::constructorTest() {
  IntersectingIOVRecordIntervalFinder finder(DummyRecord::keyForClass());
  CPPUNIT_ASSERT(finder.findingForRecords().size() == 1);
  std::set<EventSetupRecordKey> s = finder.findingForRecords();
  CPPUNIT_ASSERT(s.find(DummyRecord::keyForClass()) != s.end());
}

void testintersectingiovrecordintervalfinder::intersectionTest() {
  const EventSetupRecordKey dummyRecordKey = DummyRecord::keyForClass();

  std::vector<edm::propagate_const<std::shared_ptr<edm::EventSetupRecordIntervalFinder>>> finders;
  std::shared_ptr<DummyFinder> dummyFinder = std::make_shared<DummyFinder>();
  {
    IntersectingIOVRecordIntervalFinder intersectingFinder(dummyRecordKey);
    const edm::ValidityInterval definedInterval(edm::IOVSyncValue(edm::EventID(1, 1, 1)),
                                                edm::IOVSyncValue(edm::EventID(1, 1, 3)));
    finders.push_back(dummyFinder);
    dummyFinder->setInterval(definedInterval);
    intersectingFinder.swapFinders(finders);

    CPPUNIT_ASSERT(definedInterval ==
                   intersectingFinder.findIntervalFor(dummyRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 2))));

    const edm::ValidityInterval unknownedEndInterval(edm::IOVSyncValue(edm::EventID(1, 1, 5)),
                                                     edm::IOVSyncValue::invalidIOVSyncValue());
    dummyFinder->setInterval(unknownedEndInterval);

    CPPUNIT_ASSERT(unknownedEndInterval ==
                   intersectingFinder.findIntervalFor(dummyRecordKey, edm::IOVSyncValue(edm::EventID(1, 1, 5))));
  }

  {
    finders.clear();

    const edm::IOVSyncValue sync_1(edm::EventID(1, 1, 1));
    const edm::IOVSyncValue sync_3(edm::EventID(1, 1, 3));
    const edm::IOVSyncValue sync_4(edm::EventID(1, 1, 4));
    const edm::IOVSyncValue sync_5(edm::EventID(1, 1, 5));
    const edm::ValidityInterval definedInterval(sync_1, sync_4);
    dummyFinder->setInterval(definedInterval);
    finders.push_back(dummyFinder);

    std::shared_ptr<DummyFinder> dummyFinder2 = std::make_shared<DummyFinder>();
    dummyFinder2->setInterval(edm::ValidityInterval(sync_3, sync_5));
    finders.push_back(dummyFinder2);
    IntersectingIOVRecordIntervalFinder intersectingFinder(dummyRecordKey);
    intersectingFinder.swapFinders(finders);

    CPPUNIT_ASSERT(edm::ValidityInterval(sync_3, sync_4) == intersectingFinder.findIntervalFor(dummyRecordKey, sync_3));
  }

  {
    finders.clear();
    const edm::IOVSyncValue sync_1(edm::EventID(1, 1, 1));
    const edm::IOVSyncValue sync_3(edm::EventID(1, 1, 3));
    const edm::IOVSyncValue sync_4(edm::EventID(1, 1, 4));
    const edm::ValidityInterval definedInterval(sync_1, sync_4);
    dummyFinder->setInterval(definedInterval);
    finders.push_back(dummyFinder);

    std::shared_ptr<DummyFinder> dummyFinder2 = std::make_shared<DummyFinder>();
    dummyFinder2->setInterval(edm::ValidityInterval::invalidInterval());
    finders.push_back(dummyFinder2);
    IntersectingIOVRecordIntervalFinder intersectingFinder(dummyRecordKey);
    intersectingFinder.swapFinders(finders);

    CPPUNIT_ASSERT(edm::ValidityInterval::invalidInterval() == dummyFinder2->findIntervalFor(dummyRecordKey, sync_3));

    CPPUNIT_ASSERT(edm::ValidityInterval(sync_1, edm::IOVSyncValue::invalidIOVSyncValue()) ==
                   intersectingFinder.findIntervalFor(dummyRecordKey, sync_3));
  }

  {
    finders.clear();
    const edm::IOVSyncValue sync_1(edm::EventID(1, 1, 1));
    const edm::IOVSyncValue sync_3(edm::EventID(1, 1, 3));
    const edm::IOVSyncValue sync_4(edm::EventID(1, 1, 4));
    const edm::ValidityInterval definedInterval(sync_1, sync_4);
    dummyFinder->setInterval(definedInterval);
    finders.push_back(dummyFinder);

    std::shared_ptr<DummyFinder> dummyFinder2 = std::make_shared<DummyFinder>();
    dummyFinder2->setInterval(edm::ValidityInterval(sync_3, edm::IOVSyncValue::invalidIOVSyncValue()));
    finders.push_back(dummyFinder2);
    IntersectingIOVRecordIntervalFinder intersectingFinder(dummyRecordKey);
    intersectingFinder.swapFinders(finders);

    CPPUNIT_ASSERT(edm::ValidityInterval(sync_3, edm::IOVSyncValue::invalidIOVSyncValue()) ==
                   dummyFinder2->findIntervalFor(dummyRecordKey, sync_3));

    CPPUNIT_ASSERT(edm::ValidityInterval(sync_3, edm::IOVSyncValue::invalidIOVSyncValue()) ==
                   intersectingFinder.findIntervalFor(dummyRecordKey, sync_3));
  }

  {
    //reverse order so invalid ending is first in list
    finders.clear();
    const edm::IOVSyncValue sync_1(edm::EventID(1, 1, 1));
    const edm::IOVSyncValue sync_3(edm::EventID(1, 1, 3));
    const edm::IOVSyncValue sync_4(edm::EventID(1, 1, 4));
    const edm::ValidityInterval definedInterval(sync_1, sync_4);

    std::shared_ptr<DummyFinder> dummyFinder2 = std::make_shared<DummyFinder>();
    dummyFinder2->setInterval(edm::ValidityInterval(sync_3, edm::IOVSyncValue::invalidIOVSyncValue()));
    finders.push_back(dummyFinder2);

    dummyFinder->setInterval(definedInterval);
    finders.push_back(dummyFinder);

    IntersectingIOVRecordIntervalFinder intersectingFinder(dummyRecordKey);
    intersectingFinder.swapFinders(finders);

    CPPUNIT_ASSERT(edm::ValidityInterval(sync_3, edm::IOVSyncValue::invalidIOVSyncValue()) ==
                   dummyFinder2->findIntervalFor(dummyRecordKey, sync_3));

    CPPUNIT_ASSERT(edm::ValidityInterval(sync_3, edm::IOVSyncValue::invalidIOVSyncValue()) ==
                   intersectingFinder.findIntervalFor(dummyRecordKey, sync_3));
  }
}
