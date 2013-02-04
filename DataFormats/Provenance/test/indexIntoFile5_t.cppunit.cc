/*
 *  indexIntoFile_t.cppunit.cc
 */

#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

// This is very ugly, but I am told OK for white box  unit tests 
#define private public
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#undef private

#include <string>
#include <iostream>
#include <memory>

using namespace edm;

class TestIndexIntoFile5: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestIndexIntoFile5);  
  CPPUNIT_TEST(testDuplicateCheckerFunctions);
  CPPUNIT_TEST_SUITE_END();
  
public:

  static const IndexIntoFile::EntryType kRun = IndexIntoFile::kRun;
  static const IndexIntoFile::EntryType kLumi = IndexIntoFile::kLumi;
  static const IndexIntoFile::EntryType kEvent = IndexIntoFile::kEvent;
  static const IndexIntoFile::EntryType kEnd = IndexIntoFile::kEnd;

  class Skipped {
  public:
    Skipped(): phIndexOfSkippedEvent_(0),
               runOfSkippedEvent_(0),
               lumiOfSkippedEvent_(0),
               skippedEventEntry_(0) { }
    int phIndexOfSkippedEvent_;
    RunNumber_t runOfSkippedEvent_;
    LuminosityBlockNumber_t lumiOfSkippedEvent_;
    IndexIntoFile::EntryNumber_t skippedEventEntry_;
  };

  Skipped skipped_;
  
  void setUp() {
    // Make some fake processHistoryID's to work with
    nullPHID = ProcessHistoryID();

    ProcessConfiguration pc;
    std::unique_ptr<ProcessHistory> processHistory1(new ProcessHistory);
    ProcessHistory& ph1 = *processHistory1;
    processHistory1->push_back(pc);
    ProcessHistoryRegistry::instance()->insertMapped(ph1);
    fakePHID1 = ph1.id();

    std::unique_ptr<ProcessHistory> processHistory2(new ProcessHistory);
    ProcessHistory& ph2 = *processHistory2;
    processHistory2->push_back(pc);
    processHistory2->push_back(pc);
    ProcessHistoryRegistry::instance()->insertMapped(ph2);
    fakePHID2 = ph2.id();

    std::unique_ptr<ProcessHistory> processHistory3(new ProcessHistory);
    ProcessHistory& ph3 = *processHistory3;
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    ProcessHistoryRegistry::instance()->insertMapped(ph3);
    fakePHID3 = ph3.id();
  }

  void tearDown() { }

  void testDuplicateCheckerFunctions();

  ProcessHistoryID nullPHID;
  ProcessHistoryID fakePHID1;
  ProcessHistoryID fakePHID2;
  ProcessHistoryID fakePHID3;

  void check(edm::IndexIntoFile::IndexIntoFileItr const& iter,
             IndexIntoFile::EntryType type,
             int indexToRun,
             int indexToLumi,
             int indexToEventRange,
             long long indexToEvent,
             long long nEvents);

  // This is a helper class for IndexIntoFile.
  class TestEventFinder : public IndexIntoFile::EventFinder {
  public:
    explicit TestEventFinder() {}
    virtual ~TestEventFinder() {}
    virtual EventNumber_t getEventNumberOfEntry(IndexIntoFile::EntryNumber_t entry) const {
      return testData_.at(entry);
    }
    void push_back(EventNumber_t e) {testData_.push_back(e); }

  private:
    std::vector<EventNumber_t> testData_;
  };
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexIntoFile5);

void TestIndexIntoFile5::check(edm::IndexIntoFile::IndexIntoFileItr const& iter,
                              IndexIntoFile::EntryType type,
                              int indexToRun,
                              int indexToLumi,
                              int indexToEventRange,
                              long long indexToEvent,
                              long long nEvents) {
  bool theyMatch = iter.getEntryType() == type &&
                   iter.type() == type &&
                   iter.indexToRun() == indexToRun &&
                   iter.indexToLumi() == indexToLumi &&
                   iter.indexToEventRange() == indexToEventRange &&
                   iter.indexToEvent() == indexToEvent &&
                   iter.nEvents() == nEvents;
  if (!theyMatch) {
    std::cout << "\nExpected        " << type << "  " << indexToRun << "  "
              << indexToLumi << "  " << indexToEventRange << "  " << indexToEvent
              << "  " << nEvents << "\n";
    std::cout << "Iterator values " << iter.type() << "  " << iter.indexToRun() << "  "
              << iter.indexToLumi() << "  " << iter.indexToEventRange() << "  " << iter.indexToEvent()
              << "  " << iter.nEvents() << "\n";
  }
  CPPUNIT_ASSERT(theyMatch);
}

void TestIndexIntoFile5::testDuplicateCheckerFunctions() {

  std::set<IndexIntoFile::IndexRunLumiEventKey> relevantPreviousEvents;

  edm::IndexIntoFile indexIntoFile1;
  indexIntoFile1.addEntry(fakePHID1, 6, 1, 0, 0); // Lumi
  indexIntoFile1.addEntry(fakePHID1, 6, 0, 0, 0); // Run
  indexIntoFile1.sortVector_Run_Or_Lumi_Entries();

  //Empty Index
  edm::IndexIntoFile indexIntoFile2;
  relevantPreviousEvents.clear();
  indexIntoFile1.set_intersection(indexIntoFile2, relevantPreviousEvents);
  CPPUNIT_ASSERT(relevantPreviousEvents.empty());

  relevantPreviousEvents.clear();
  indexIntoFile2.set_intersection(indexIntoFile1, relevantPreviousEvents);
  CPPUNIT_ASSERT(relevantPreviousEvents.empty());

  // Run ranges do not overlap
  edm::IndexIntoFile indexIntoFile3;
  indexIntoFile3.addEntry(fakePHID1, 7, 0, 0, 0); // Run
  indexIntoFile3.sortVector_Run_Or_Lumi_Entries();

  relevantPreviousEvents.clear();
  indexIntoFile1.set_intersection(indexIntoFile3, relevantPreviousEvents);
  CPPUNIT_ASSERT(relevantPreviousEvents.empty());

  relevantPreviousEvents.clear();
  indexIntoFile3.set_intersection(indexIntoFile1, relevantPreviousEvents);
  CPPUNIT_ASSERT(relevantPreviousEvents.empty());

  // No lumis
  edm::IndexIntoFile indexIntoFile4;
  indexIntoFile4.addEntry(fakePHID1, 6, 0, 0, 0); // Run
  indexIntoFile4.addEntry(fakePHID1, 7, 0, 0, 0); // Run
  indexIntoFile4.sortVector_Run_Or_Lumi_Entries();

  relevantPreviousEvents.clear();
  indexIntoFile1.set_intersection(indexIntoFile4, relevantPreviousEvents);
  CPPUNIT_ASSERT(relevantPreviousEvents.empty());

  relevantPreviousEvents.clear();
  indexIntoFile4.set_intersection(indexIntoFile1, relevantPreviousEvents);
  CPPUNIT_ASSERT(relevantPreviousEvents.empty());

  // Lumi ranges do not overlap
  edm::IndexIntoFile indexIntoFile5;
  indexIntoFile5.addEntry(fakePHID1, 6, 2, 0, 0); // Lumi
  indexIntoFile5.addEntry(fakePHID1, 6, 0, 0, 0); // Run
  indexIntoFile5.addEntry(fakePHID1, 6, 0, 0, 0); // Run
  indexIntoFile5.sortVector_Run_Or_Lumi_Entries();

  relevantPreviousEvents.clear();
  indexIntoFile1.set_intersection(indexIntoFile5, relevantPreviousEvents);
  CPPUNIT_ASSERT(relevantPreviousEvents.empty());

  relevantPreviousEvents.clear();
  indexIntoFile5.set_intersection(indexIntoFile1, relevantPreviousEvents);
  CPPUNIT_ASSERT(relevantPreviousEvents.empty());


  for (int j = 0; j < 2; ++j) {
    edm::IndexIntoFile indexIntoFile11;
    indexIntoFile11.addEntry(fakePHID1, 6, 2, 0, 0); // Lumi
    indexIntoFile11.addEntry(fakePHID1, 6, 3, 0, 1); // Lumi
    indexIntoFile11.addEntry(fakePHID1, 6, 3, 0, 2); // Lumi
    indexIntoFile11.addEntry(fakePHID1, 6, 0, 0, 0); // Run
    indexIntoFile11.addEntry(fakePHID1, 6, 0, 0, 1); // Run
    indexIntoFile11.addEntry(fakePHID1, 7, 1, 1, 0); // Event
    indexIntoFile11.addEntry(fakePHID1, 7, 1, 2, 1); // Event
    indexIntoFile11.addEntry(fakePHID1, 7, 1, 3, 2); // Event
    indexIntoFile11.addEntry(fakePHID1, 7, 1, 4, 3); // Event
    indexIntoFile11.addEntry(fakePHID1, 7, 1, 0, 3); // Lumi
    indexIntoFile11.addEntry(fakePHID1, 7, 0, 0, 2); // Run
    indexIntoFile11.addEntry(fakePHID1, 8, 1, 1, 4); // Event
    indexIntoFile11.addEntry(fakePHID1, 8, 1, 0, 4); // Lumi
    indexIntoFile11.addEntry(fakePHID1, 8, 0, 0, 3); // Run
    indexIntoFile11.addEntry(fakePHID1, 8, 1, 2, 5); // Event
    indexIntoFile11.addEntry(fakePHID1, 8, 1, 0, 5); // Lumi
    indexIntoFile11.addEntry(fakePHID1, 8, 0, 0, 4); // Run
    indexIntoFile11.sortVector_Run_Or_Lumi_Entries();
   
    edm::IndexIntoFile indexIntoFile12;
    indexIntoFile12.addEntry(fakePHID1, 6, 1, 0, 0); // Lumi
    indexIntoFile12.addEntry(fakePHID1, 6, 3, 0, 1); // Lumi
    indexIntoFile12.addEntry(fakePHID1, 6, 3, 0, 2); // Lumi
    indexIntoFile12.addEntry(fakePHID1, 6, 0, 0, 0); // Run
    indexIntoFile12.addEntry(fakePHID1, 6, 0, 0, 1); // Run
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 1, 0); // Event
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 7, 1); // Event
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 3, 2); // Event
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 8, 3); // Event
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 0, 3); // Lumi
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 11, 4); // Event
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 6, 5); // Event
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 1, 6); // Event
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 4, 7); // Event
    indexIntoFile12.addEntry(fakePHID1, 7, 1, 0, 4); // Lumi
    indexIntoFile12.addEntry(fakePHID1, 7, 0, 0, 2); // Run
    indexIntoFile12.sortVector_Run_Or_Lumi_Entries();

    edm::IndexIntoFile indexIntoFile22;
    indexIntoFile22.addEntry(fakePHID1, 6, 1, 0, 0); // Lumi
    indexIntoFile22.addEntry(fakePHID1, 6, 3, 0, 1); // Lumi
    indexIntoFile22.addEntry(fakePHID1, 6, 3, 0, 2); // Lumi
    indexIntoFile22.addEntry(fakePHID1, 6, 0, 0, 0); // Run
    indexIntoFile22.addEntry(fakePHID1, 6, 0, 0, 1); // Run
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 11, 0); // Event
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 7, 1); // Event
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 3, 2); // Event
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 8, 3); // Event
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 0, 3); // Lumi
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 11, 4); // Event
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 6, 5); // Event
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 1, 6); // Event
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 4, 7); // Event
    indexIntoFile22.addEntry(fakePHID1, 7, 1, 0, 4); // Lumi
    indexIntoFile22.addEntry(fakePHID1, 7, 0, 0, 2); // Run
    indexIntoFile22.sortVector_Run_Or_Lumi_Entries();

    TestEventFinder* ptr11(new TestEventFinder);
    ptr11->push_back(1);
    ptr11->push_back(2);
    ptr11->push_back(3);
    ptr11->push_back(4);
    ptr11->push_back(1);
    ptr11->push_back(2);

    boost::shared_ptr<IndexIntoFile::EventFinder> shptr11(ptr11);
    indexIntoFile11.setEventFinder(shptr11);

    TestEventFinder* ptr12(new TestEventFinder);
    ptr12->push_back(1);
    ptr12->push_back(7);
    ptr12->push_back(3);
    ptr12->push_back(8);
    ptr12->push_back(11);
    ptr12->push_back(6);
    ptr12->push_back(1);
    ptr12->push_back(4);

    boost::shared_ptr<IndexIntoFile::EventFinder> shptr12(ptr12);
    indexIntoFile12.setEventFinder(shptr12);

    TestEventFinder* ptr22(new TestEventFinder);
    ptr22->push_back(11);
    ptr22->push_back(7);
    ptr22->push_back(3);
    ptr22->push_back(8);
    ptr22->push_back(11);
    ptr22->push_back(6);
    ptr22->push_back(1);
    ptr22->push_back(4);

    boost::shared_ptr<IndexIntoFile::EventFinder> shptr22(ptr22);
    indexIntoFile22.setEventFinder(shptr22);

    if (j == 0) {
      indexIntoFile11.fillEventNumbers();
      indexIntoFile12.fillEventNumbers();
      indexIntoFile22.fillEventNumbers();
    }
    else {
      indexIntoFile11.fillEventEntries();
      indexIntoFile12.fillEventEntries();
      indexIntoFile22.fillEventEntries();
    }

    CPPUNIT_ASSERT(!indexIntoFile11.containsDuplicateEvents());
    CPPUNIT_ASSERT(indexIntoFile12.containsDuplicateEvents());
    CPPUNIT_ASSERT(indexIntoFile22.containsDuplicateEvents());

    relevantPreviousEvents.clear();
    indexIntoFile11.set_intersection(indexIntoFile12, relevantPreviousEvents);
    CPPUNIT_ASSERT(relevantPreviousEvents.size() == 3);
    std::set<IndexIntoFile::IndexRunLumiEventKey>::const_iterator iter = relevantPreviousEvents.begin();
    CPPUNIT_ASSERT(iter->event() == 1);
    CPPUNIT_ASSERT(iter->processHistoryIDIndex() == 0);
    CPPUNIT_ASSERT(iter->run() == 7);
    CPPUNIT_ASSERT(iter->lumi() == 1);
    ++iter;
    CPPUNIT_ASSERT(iter->event() == 3);
    ++iter;
    CPPUNIT_ASSERT(iter->event() == 4);
    
    relevantPreviousEvents.clear();
    indexIntoFile12.set_intersection(indexIntoFile11, relevantPreviousEvents);
    CPPUNIT_ASSERT(relevantPreviousEvents.size() == 3);
    iter = relevantPreviousEvents.begin();
    CPPUNIT_ASSERT(iter->event() == 1);
    CPPUNIT_ASSERT(iter->processHistoryIDIndex() == 0);
    CPPUNIT_ASSERT(iter->run() == 7);
    CPPUNIT_ASSERT(iter->lumi() == 1);
    ++iter;
    CPPUNIT_ASSERT(iter->event() == 3);
    ++iter;
    CPPUNIT_ASSERT(iter->event() == 4);
    
  }
}
