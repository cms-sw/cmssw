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

class TestIndexIntoFile: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestIndexIntoFile);  
  CPPUNIT_TEST(testRunOrLumiEntry);
  CPPUNIT_TEST(testRunOrLumiIndexes);
  CPPUNIT_TEST(testEventEntry);
  CPPUNIT_TEST(testSortedRunOrLumiItr);
  CPPUNIT_TEST(testKeys);
  CPPUNIT_TEST(testConstructor);
  CPPUNIT_TEST(testAddEntryAndFixAndSort);
  CPPUNIT_TEST(testEmptyIndex);
  CPPUNIT_TEST(testIterEndWithEvent);
  CPPUNIT_TEST(testIterEndWithLumi);
  CPPUNIT_TEST(testIterEndWithRun);
  CPPUNIT_TEST(testIterLastLumiRangeNoEvents);
  CPPUNIT_TEST(testSkip);
  CPPUNIT_TEST(testSkip2);
  CPPUNIT_TEST(testSkip3);
  CPPUNIT_TEST(testFind);
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
    std::auto_ptr<ProcessHistory> processHistory1(new ProcessHistory);
    ProcessHistory& ph1 = *processHistory1;
    processHistory1->push_back(pc);
    ProcessHistoryRegistry::instance()->insertMapped(ph1);
    fakePHID1 = ph1.id();

    std::auto_ptr<ProcessHistory> processHistory2(new ProcessHistory);
    ProcessHistory& ph2 = *processHistory2;
    processHistory2->push_back(pc);
    processHistory2->push_back(pc);
    ProcessHistoryRegistry::instance()->insertMapped(ph2);
    fakePHID2 = ph2.id();

    std::auto_ptr<ProcessHistory> processHistory3(new ProcessHistory);
    ProcessHistory& ph3 = *processHistory3;
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    ProcessHistoryRegistry::instance()->insertMapped(ph3);
    fakePHID3 = ph3.id();
  }

  void tearDown() { }

  void testRunOrLumiEntry();
  void testRunOrLumiIndexes();
  void testEventEntry();
  void testSortedRunOrLumiItr();
  void testKeys();
  void testConstructor();
  void testAddEntryAndFixAndSort();
  void testEmptyIndex();
  void testIterEndWithEvent();
  void testIterEndWithLumi();
  void testIterEndWithRun();
  void testIterLastLumiRangeNoEvents();
  void testSkip();
  void testSkip2();
  void testSkip3();
  void testFind();
  void testDuplicateCheckerFunctions();
  void testReduce();

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

  void skipEventForward(edm::IndexIntoFile::IndexIntoFileItr & iter);
  void skipEventBackward(edm::IndexIntoFile::IndexIntoFileItr & iter);
  void checkSkipped(int phIndexOfSkippedEvent,
                    RunNumber_t runOfSkippedEvent,
                    LuminosityBlockNumber_t lumiOfSkippedEvent,
                    IndexIntoFile::EntryNumber_t skippedEventEntry);

  void checkIDRunLumiEntry(edm::IndexIntoFile::IndexIntoFileItr const& iter,
                           int phIndex,
                           RunNumber_t run,
                           LuminosityBlockNumber_t lumi,
                           IndexIntoFile::EntryNumber_t entry);

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
CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexIntoFile);

void TestIndexIntoFile::check(edm::IndexIntoFile::IndexIntoFileItr const& iter,
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

void TestIndexIntoFile::checkSkipped(int phIndexOfSkippedEvent,
                                     RunNumber_t runOfSkippedEvent,
                                     LuminosityBlockNumber_t lumiOfSkippedEvent,
                                     IndexIntoFile::EntryNumber_t skippedEventEntry) {
  bool theyMatch = skipped_.phIndexOfSkippedEvent_ == phIndexOfSkippedEvent &&
                   skipped_.runOfSkippedEvent_ == runOfSkippedEvent &&
                   skipped_.lumiOfSkippedEvent_ == lumiOfSkippedEvent &&
                   skipped_.skippedEventEntry_ == skippedEventEntry;



  if (!theyMatch) {
    std::cout << "\nExpected        " << phIndexOfSkippedEvent << "  " << runOfSkippedEvent << "  "
              << lumiOfSkippedEvent << "  " << skippedEventEntry << "\n";
    std::cout << "Actual          " << skipped_.phIndexOfSkippedEvent_ << "  " << skipped_.runOfSkippedEvent_ << "  "
              << skipped_.lumiOfSkippedEvent_ << "  " << skipped_.skippedEventEntry_ << "\n";
  }
  CPPUNIT_ASSERT(theyMatch);
}

void TestIndexIntoFile::checkIDRunLumiEntry(edm::IndexIntoFile::IndexIntoFileItr const& iter,
                                            int phIndex,
                                            RunNumber_t run,
                                            LuminosityBlockNumber_t lumi,
                                            IndexIntoFile::EntryNumber_t entry) {
  bool theyMatch = iter.processHistoryIDIndex() == phIndex &&
                   iter.run() == run &&
                   iter.lumi() == lumi &&
                   iter.entry() == entry;

  if (!theyMatch) {
    std::cout << "\nExpected        " << phIndex << "  " << run << "  "
              << lumi << "  " << entry << "\n";
    std::cout << "Actual          " << iter.processHistoryIDIndex() << "  " << iter.run() << "  "
              << iter.lumi() << "  " << iter.entry() << "\n";
  }
  CPPUNIT_ASSERT(theyMatch);
}

void TestIndexIntoFile::skipEventForward(edm::IndexIntoFile::IndexIntoFileItr & iter) {
  iter.skipEventForward(skipped_.phIndexOfSkippedEvent_,
                        skipped_.runOfSkippedEvent_,
                        skipped_.lumiOfSkippedEvent_,
                        skipped_.skippedEventEntry_);
}

void TestIndexIntoFile::skipEventBackward(edm::IndexIntoFile::IndexIntoFileItr & iter) {
  iter.skipEventBackward(skipped_.phIndexOfSkippedEvent_,
                         skipped_.runOfSkippedEvent_,
                         skipped_.lumiOfSkippedEvent_,
                         skipped_.skippedEventEntry_);
}



void TestIndexIntoFile::testRunOrLumiEntry() {

  edm::IndexIntoFile::RunOrLumiEntry r1;
  CPPUNIT_ASSERT(r1.orderPHIDRun() == edm::IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(r1.orderPHIDRunLumi() == edm::IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(r1.entry() == edm::IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(r1.processHistoryIDIndex() == edm::IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(r1.run() == edm::IndexIntoFile::invalidRun);
  CPPUNIT_ASSERT(r1.lumi() == edm::IndexIntoFile::invalidLumi);
  CPPUNIT_ASSERT(r1.beginEvents() == edm::IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(r1.endEvents() == edm::IndexIntoFile::invalidEntry);

  edm::IndexIntoFile::RunOrLumiEntry r2(1, 2, 3, 4, 5, 6, 7, 8);

  CPPUNIT_ASSERT(r2.orderPHIDRun() == 1);
  CPPUNIT_ASSERT(r2.orderPHIDRunLumi() == 2);
  CPPUNIT_ASSERT(r2.entry() == 3);
  CPPUNIT_ASSERT(r2.processHistoryIDIndex() == 4);
  CPPUNIT_ASSERT(r2.run() == 5);
  CPPUNIT_ASSERT(r2.lumi() == 6);
  CPPUNIT_ASSERT(r2.beginEvents() == 7);
  CPPUNIT_ASSERT(r2.endEvents() == 8);

  CPPUNIT_ASSERT(r2.isRun() == false);

  edm::IndexIntoFile::RunOrLumiEntry r3 (1, 2, 3, 4, 5, edm::IndexIntoFile::invalidLumi, 7, 8);

  CPPUNIT_ASSERT(r3.isRun() == true);

  r3.setOrderPHIDRun(11);
  CPPUNIT_ASSERT(r3.orderPHIDRun() == 11);
  r3.setProcessHistoryIDIndex(12);
  CPPUNIT_ASSERT(r3.processHistoryIDIndex() == 12);
  r3.setOrderPHIDRun(1);

  CPPUNIT_ASSERT(!(r2 < r3));
  CPPUNIT_ASSERT(!(r3 < r2));
  
  edm::IndexIntoFile::RunOrLumiEntry r4 (10, 1, 1, 4, 5, 6, 7, 8);
  CPPUNIT_ASSERT(r2 < r4);
  CPPUNIT_ASSERT(!(r4 < r2));

  edm::IndexIntoFile::RunOrLumiEntry r5 (1, 10, 1, 4, 5, 6, 7, 8);
  CPPUNIT_ASSERT(r2 < r5);
  CPPUNIT_ASSERT(!(r5 < r2));

  edm::IndexIntoFile::RunOrLumiEntry r6 (1, 2, 10, 4, 5, 6, 7, 8);
  CPPUNIT_ASSERT(r2 < r6);
  CPPUNIT_ASSERT(!(r6 < r2));

  r3.setOrderPHIDRunLumi(1001);
  CPPUNIT_ASSERT(r3.orderPHIDRunLumi() == 1001);
}

void TestIndexIntoFile::testRunOrLumiIndexes() {

  edm::IndexIntoFile::RunOrLumiIndexes r1(1, 2, 3, 4);
  CPPUNIT_ASSERT(r1.processHistoryIDIndex() == 1);
  CPPUNIT_ASSERT(r1.run() == 2);
  CPPUNIT_ASSERT(r1.lumi() == 3);
  CPPUNIT_ASSERT(r1.indexToGetEntry() == 4);
  CPPUNIT_ASSERT(r1.beginEventNumbers() == -1);
  CPPUNIT_ASSERT(r1.endEventNumbers() == -1);

  r1.setBeginEventNumbers(11);
  r1.setEndEventNumbers(12);
  CPPUNIT_ASSERT(r1.beginEventNumbers() == 11);
  CPPUNIT_ASSERT(r1.endEventNumbers() == 12);

  CPPUNIT_ASSERT(r1.isRun() == false);

  edm::IndexIntoFile::RunOrLumiIndexes r2(1, 2, edm::IndexIntoFile::invalidLumi, 4);
  CPPUNIT_ASSERT(r2.isRun() == true);
  
  edm::IndexIntoFile::RunOrLumiIndexes r3(1, 2, 3, 4);
  CPPUNIT_ASSERT(!(r1 < r3));
  CPPUNIT_ASSERT(!(r3 < r1));

  edm::IndexIntoFile::RunOrLumiIndexes r4(11, 2, 3, 4);
  CPPUNIT_ASSERT(r1 < r4);
  CPPUNIT_ASSERT(!(r4 < r1));

  edm::IndexIntoFile::RunOrLumiIndexes r5(1, 11, 1, 4);
  CPPUNIT_ASSERT(r1 < r5);
  CPPUNIT_ASSERT(!(r5 < r1));

  edm::IndexIntoFile::RunOrLumiIndexes r6(1, 2, 11, 4);
  CPPUNIT_ASSERT(r1 < r6);
  CPPUNIT_ASSERT(!(r6 < r1));

  Compare_Index_Run c;
  CPPUNIT_ASSERT(!c(r1, r6));
  CPPUNIT_ASSERT(!c(r6, r1));
  CPPUNIT_ASSERT(c(r1, r5));
  CPPUNIT_ASSERT(!c(r5, r1));
  CPPUNIT_ASSERT(c(r1, r4));
  CPPUNIT_ASSERT(!c(r4, r1));

  Compare_Index c1;
  CPPUNIT_ASSERT(!c1(r1, r5));
  CPPUNIT_ASSERT(!c1(r5, r1));
  CPPUNIT_ASSERT(c1(r1, r4));
  CPPUNIT_ASSERT(!c1(r4, r1));
}

void TestIndexIntoFile::testEventEntry() {
  edm::IndexIntoFile::EventEntry e1;
  CPPUNIT_ASSERT(e1.event() == edm::IndexIntoFile::invalidEvent);
  CPPUNIT_ASSERT(e1.entry() == edm::IndexIntoFile::invalidEntry);

  edm::IndexIntoFile::EventEntry e2(100, 200);
  CPPUNIT_ASSERT(e2.event() == 100);
  CPPUNIT_ASSERT(e2.entry() == 200);

  edm::IndexIntoFile::EventEntry e3(100, 300);
  edm::IndexIntoFile::EventEntry e4(200, 300);
  edm::IndexIntoFile::EventEntry e5(200, 100);

  CPPUNIT_ASSERT(e2 == e3);
  CPPUNIT_ASSERT(!(e3 == e4));

  CPPUNIT_ASSERT(e3 < e4);
  CPPUNIT_ASSERT(e3 < e5);
  CPPUNIT_ASSERT(!(e4 < e5));
}

void TestIndexIntoFile::testSortedRunOrLumiItr() {

  edm::IndexIntoFile indexIntoFile0;
  CPPUNIT_ASSERT(indexIntoFile0.empty());
  edm::IndexIntoFile::SortedRunOrLumiItr iter(&indexIntoFile0, 0);
  CPPUNIT_ASSERT(iter.indexIntoFile() == &indexIntoFile0);
  CPPUNIT_ASSERT(iter.runOrLumi() == 0);
  ++iter;
  CPPUNIT_ASSERT(iter.runOrLumi() == 0);
  CPPUNIT_ASSERT(iter == indexIntoFile0.beginRunOrLumi());
  CPPUNIT_ASSERT(iter == indexIntoFile0.endRunOrLumi());

  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 1, 1, 0); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 2, 1); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 0); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 2, 1, 2); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 2, 3); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 3, 4); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 1); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 1, 3, 5); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 4, 6); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 0); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0); // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  CPPUNIT_ASSERT(!indexIntoFile.empty());

  unsigned count = 0;
  IndexIntoFile::SortedRunOrLumiItr runOrLumi = indexIntoFile.beginRunOrLumi();
  for (IndexIntoFile::SortedRunOrLumiItr endRunOrLumi = indexIntoFile.endRunOrLumi();
       runOrLumi != endRunOrLumi; ++runOrLumi) {

    long long beginEventNumbers;
    long long endEventNumbers;
    IndexIntoFile::EntryNumber_t beginEventEntry;
    IndexIntoFile::EntryNumber_t endEventEntry;
    runOrLumi.getRange(beginEventNumbers, endEventNumbers, beginEventEntry, endEventEntry);

    if (count == 0) {
      CPPUNIT_ASSERT(runOrLumi.isRun());
      CPPUNIT_ASSERT(beginEventNumbers == -1);
      CPPUNIT_ASSERT(endEventNumbers == -1);
      CPPUNIT_ASSERT(beginEventEntry == -1);
      CPPUNIT_ASSERT(endEventEntry == -1);
    }
    else if (count == 3) {
      CPPUNIT_ASSERT(!runOrLumi.isRun());
      CPPUNIT_ASSERT(beginEventNumbers == 4);
      CPPUNIT_ASSERT(endEventNumbers == 7);
      CPPUNIT_ASSERT(beginEventEntry == 2);
      CPPUNIT_ASSERT(endEventEntry == 5);

      IndexIntoFile::RunOrLumiIndexes const& indexes = runOrLumi.runOrLumiIndexes();
      CPPUNIT_ASSERT(indexes.processHistoryIDIndex() == 0);
      CPPUNIT_ASSERT(indexes.run() == 1U);
      CPPUNIT_ASSERT(indexes.lumi() == 2U);
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 3);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == 4);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == 7);
    }

    CPPUNIT_ASSERT(runOrLumi.runOrLumi() == count);
    ++count;
  }
  CPPUNIT_ASSERT(count == 4U);
  CPPUNIT_ASSERT(runOrLumi.runOrLumi() == 4U);
  ++runOrLumi;
  CPPUNIT_ASSERT(runOrLumi.runOrLumi() == 4U);

  CPPUNIT_ASSERT(runOrLumi == indexIntoFile.endRunOrLumi());
  CPPUNIT_ASSERT(!(runOrLumi == indexIntoFile.beginRunOrLumi()));
  CPPUNIT_ASSERT(!(iter == indexIntoFile.beginRunOrLumi()));

  CPPUNIT_ASSERT(!(runOrLumi != indexIntoFile.endRunOrLumi()));
  CPPUNIT_ASSERT(runOrLumi != indexIntoFile.beginRunOrLumi());
  CPPUNIT_ASSERT(iter != indexIntoFile.beginRunOrLumi());
}

void TestIndexIntoFile::testKeys() {
  IndexIntoFile::IndexRunKey key1(1, 2);
  CPPUNIT_ASSERT(key1.processHistoryIDIndex() == 1);
  CPPUNIT_ASSERT(key1.run() == 2);

  IndexIntoFile::IndexRunKey key2(1, 2);
  CPPUNIT_ASSERT(!(key1 < key2));
  CPPUNIT_ASSERT(!(key2 < key1));

  IndexIntoFile::IndexRunKey key3(1, 3);
  CPPUNIT_ASSERT(key1 < key3);
  CPPUNIT_ASSERT(!(key3 < key1));

  IndexIntoFile::IndexRunKey key4(10, 1);
  CPPUNIT_ASSERT(key1 < key4);
  CPPUNIT_ASSERT(!(key4 < key1));

  IndexIntoFile::IndexRunLumiKey k1(1, 2, 3);
  CPPUNIT_ASSERT(k1.processHistoryIDIndex() == 1);
  CPPUNIT_ASSERT(k1.run() == 2);
  CPPUNIT_ASSERT(k1.lumi() == 3);

  IndexIntoFile::IndexRunLumiKey k2(1, 2, 3);
  CPPUNIT_ASSERT(!(k1 < k2));
  CPPUNIT_ASSERT(!(k2 < k1));

  IndexIntoFile::IndexRunLumiKey k3(1, 2, 4);
  CPPUNIT_ASSERT(k1 < k3);
  CPPUNIT_ASSERT(!(k3 < k1));

  IndexIntoFile::IndexRunLumiKey k4(1, 3, 1);
  CPPUNIT_ASSERT(k1 < k4);
  CPPUNIT_ASSERT(!(k4 < k1));

  IndexIntoFile::IndexRunLumiKey k5(11, 1, 1);
  CPPUNIT_ASSERT(k1 < k5);
  CPPUNIT_ASSERT(!(k5 < k1));

  IndexIntoFile::IndexRunLumiEventKey e1(1, 2, 3, 4);
  CPPUNIT_ASSERT(e1.processHistoryIDIndex() == 1);
  CPPUNIT_ASSERT(e1.run() == 2);
  CPPUNIT_ASSERT(e1.lumi() == 3);
  CPPUNIT_ASSERT(e1.event() == 4);

  IndexIntoFile::IndexRunLumiEventKey e2(1, 2, 3, 4);
  CPPUNIT_ASSERT(!(e1 < e2));
  CPPUNIT_ASSERT(!(e2 < e1));

  IndexIntoFile::IndexRunLumiEventKey e3(1, 2, 3, 5);
  CPPUNIT_ASSERT(e1 < e3);
  CPPUNIT_ASSERT(!(e3 < e1));

  IndexIntoFile::IndexRunLumiEventKey e4(1, 2, 11, 1);
  CPPUNIT_ASSERT(e1 < e4);
  CPPUNIT_ASSERT(!(e4 < e1));

  IndexIntoFile::IndexRunLumiEventKey e5(1, 11, 1, 1);
  CPPUNIT_ASSERT(e1 < e5);
  CPPUNIT_ASSERT(!(e5 < e1));

  IndexIntoFile::IndexRunLumiEventKey e6(11, 1, 1, 1);
  CPPUNIT_ASSERT(e1 < e6);
  CPPUNIT_ASSERT(!(e6 < e1));
}

void TestIndexIntoFile::testConstructor() {
  edm::IndexIntoFile indexIntoFile;
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries().empty());
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs().empty());
  CPPUNIT_ASSERT(indexIntoFile.eventEntries().empty());
  CPPUNIT_ASSERT(indexIntoFile.eventNumbers().empty());
  CPPUNIT_ASSERT(indexIntoFile.setRunOrLumiEntries().empty());
  CPPUNIT_ASSERT(indexIntoFile.setProcessHistoryIDs().empty());
}

void TestIndexIntoFile::testAddEntryAndFixAndSort() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.fillEventNumbersOrEntries(true, true); // Should do nothing, it is empty at this point

  indexIntoFile.addEntry(fakePHID1, 11, 12, 7, 0); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 6, 1); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 0, 0); // Lumi
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries().size() == 1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[0].processHistoryIDIndex() == 0);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs().size() == 1);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[0].orderPHIDRun() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[0].orderPHIDRunLumi() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[0].beginEvents() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[0].endEvents() == 2);

  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 0); // Run
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries().size() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[1].processHistoryIDIndex() == 0);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs().size() == 1);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[1].orderPHIDRun() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[1].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[1].beginEvents() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[1].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID2, 11, 12, 10, 2); // Event
  indexIntoFile.addEntry(fakePHID2, 11, 12, 9, 3); // Event
  indexIntoFile.addEntry(fakePHID2, 11, 12, 0, 1); // Lumi
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries().size() == 3);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[2].processHistoryIDIndex() == 1);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs().size() == 2);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs()[1] == fakePHID2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[2].orderPHIDRun() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.setRunOrLumiEntries()[2].orderPHIDRunLumi() == 1);
  CPPUNIT_ASSERT(indexIntoFile.setRunOrLumiEntries()[2].beginEvents() == 2);
  CPPUNIT_ASSERT(indexIntoFile.setRunOrLumiEntries()[2].endEvents() == 4);

  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 1); // Run
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries().size() == 4);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[3].processHistoryIDIndex() == 1);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs().size() == 2);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs()[1] == fakePHID2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[3].orderPHIDRun() == 1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[3].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[3].beginEvents() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[3].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID1, 11, 12, 5, 4); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 4, 5); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 0, 2); // Lumi
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries().size() == 5);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].processHistoryIDIndex() == 0);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs().size() == 2);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].orderPHIDRun() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].orderPHIDRunLumi() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].beginEvents() == 4);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].endEvents() == 6);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].entry() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].run() == 11);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].lumi() == 12);

  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 2); // Run
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries().size() == 6);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs().size() == 2);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  IndexIntoFile::RunOrLumiEntry const& runOrLumiEntry = indexIntoFile.runOrLumiEntries()[5];
  CPPUNIT_ASSERT(runOrLumiEntry.processHistoryIDIndex() == 0);
  CPPUNIT_ASSERT(runOrLumiEntry.orderPHIDRun() == 0);
  CPPUNIT_ASSERT(runOrLumiEntry.orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(runOrLumiEntry.beginEvents() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(runOrLumiEntry.endEvents() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(runOrLumiEntry.entry() == 2);
  CPPUNIT_ASSERT(runOrLumiEntry.run() == 11);
  CPPUNIT_ASSERT(runOrLumiEntry.lumi() == IndexIntoFile::invalidLumi);

  indexIntoFile.addEntry(fakePHID1, 1, 3, 0, 3); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 2, 8, 6); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 4); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 5); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 3); // Run

  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[0].orderPHIDRun() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[1].orderPHIDRun() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[2].orderPHIDRun() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[3].orderPHIDRun() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].orderPHIDRun() == 1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[5].orderPHIDRun() == 1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[6].orderPHIDRun() == 3);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[7].orderPHIDRun() == 3);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[8].orderPHIDRun() == 3);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[9].orderPHIDRun() == 3);

  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[0].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[1].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[2].orderPHIDRunLumi() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[3].orderPHIDRunLumi() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[5].orderPHIDRunLumi() == 1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[6].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[7].orderPHIDRunLumi() == 3);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[8].orderPHIDRunLumi() == 4);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[9].orderPHIDRunLumi() == 4);

  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[0].entry() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[1].entry() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[2].entry() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[3].entry() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].entry() == 1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[5].entry() == 1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[6].entry() == 3);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[7].entry() == 3);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[8].entry() == 4);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[9].entry() == 5);

  CPPUNIT_ASSERT(indexIntoFile.processHistoryID(0) == fakePHID1);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryID(1) == fakePHID2);

  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs()[1] == fakePHID2);

  std::vector<ProcessHistoryID> sourcePHIDs;
  sourcePHIDs.push_back(fakePHID3);
  sourcePHIDs.push_back(fakePHID2);
  
  indexIntoFile.fixIndexes(sourcePHIDs);

  CPPUNIT_ASSERT(sourcePHIDs.size() == 3);
  CPPUNIT_ASSERT(sourcePHIDs[0] == fakePHID3);
  CPPUNIT_ASSERT(sourcePHIDs[1] == fakePHID2);
  CPPUNIT_ASSERT(sourcePHIDs[2] == fakePHID1);
  CPPUNIT_ASSERT(sourcePHIDs == indexIntoFile.processHistoryIDs());

  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[0].processHistoryIDIndex() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[1].processHistoryIDIndex() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[2].processHistoryIDIndex() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[3].processHistoryIDIndex() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[4].processHistoryIDIndex() == 1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[5].processHistoryIDIndex() == 1);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[6].processHistoryIDIndex() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[7].processHistoryIDIndex() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[8].processHistoryIDIndex() == 2);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries()[9].processHistoryIDIndex() == 2);

  std::vector<ProcessHistoryID>& phids = indexIntoFile.setProcessHistoryIDs();
  phids.push_back(nullPHID);
  CPPUNIT_ASSERT(nullPHID == indexIntoFile.processHistoryID(3));
 

  unsigned count = 0;
  IndexIntoFile::SortedRunOrLumiItr runOrLumi = indexIntoFile.beginRunOrLumi();
  for (IndexIntoFile::SortedRunOrLumiItr endRunOrLumi = indexIntoFile.endRunOrLumi();
       runOrLumi != endRunOrLumi; ++runOrLumi, ++count) {

    IndexIntoFile::RunOrLumiIndexes const& indexes = runOrLumi.runOrLumiIndexes();
    if (count == 0) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 4);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == -1);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == -1);
    }
    if (count == 1) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 5);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == 0);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == 2);
    }
    if (count == 2) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 6);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == -1);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == -1);
    }
    if (count == 3) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 8);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == 2);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == 3);
    }
    if (count == 4) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 9);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == 2);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == 3);
    }
    if (count == 5) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 7);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == 3);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == 3);
    }
    if (count == 6) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 0);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == -1);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == -1);
    }
    if (count == 7) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 1);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == -1);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == -1);
    }
    if (count == 8) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 2);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == 3);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == 7);
    }
    if (count == 9) {
      CPPUNIT_ASSERT(indexes.indexToGetEntry() == 3);
      CPPUNIT_ASSERT(indexes.beginEventNumbers() == 3);
      CPPUNIT_ASSERT(indexes.endEventNumbers() == 7);
    }
  }

  std::vector<EventNumber_t>& eventNumbers = indexIntoFile.eventNumbers();

  eventNumbers.push_back(10);
  eventNumbers.push_back(9);
  eventNumbers.push_back(8);
  eventNumbers.push_back(7);
  eventNumbers.push_back(6);
  eventNumbers.push_back(5);
  eventNumbers.push_back(4);
  indexIntoFile.sortEvents();

  CPPUNIT_ASSERT(eventNumbers[0] == 9);
  CPPUNIT_ASSERT(eventNumbers[1] == 10);
  CPPUNIT_ASSERT(eventNumbers[2] == 8);
  CPPUNIT_ASSERT(eventNumbers[3] == 4);
  CPPUNIT_ASSERT(eventNumbers[4] == 5);
  CPPUNIT_ASSERT(eventNumbers[5] == 6);
  CPPUNIT_ASSERT(eventNumbers[6] == 7);

  std::vector<IndexIntoFile::EventEntry>&  eventEntries  = indexIntoFile.eventEntries();

  eventEntries.push_back(IndexIntoFile::EventEntry(10, 2));
  eventEntries.push_back(IndexIntoFile::EventEntry(9, 3));
  eventEntries.push_back(IndexIntoFile::EventEntry(8, 6));
  eventEntries.push_back(IndexIntoFile::EventEntry(7, 0));
  eventEntries.push_back(IndexIntoFile::EventEntry(6, 1));
  eventEntries.push_back(IndexIntoFile::EventEntry(5, 4));
  eventEntries.push_back(IndexIntoFile::EventEntry(4, 5));
  indexIntoFile.sortEventEntries();

  CPPUNIT_ASSERT(eventEntries[0].event() == 9);
  CPPUNIT_ASSERT(eventEntries[1].event() == 10);
  CPPUNIT_ASSERT(eventEntries[2].event() == 8);
  CPPUNIT_ASSERT(eventEntries[3].event() == 4);
  CPPUNIT_ASSERT(eventEntries[4].event() == 5);
  CPPUNIT_ASSERT(eventEntries[5].event() == 6);
  CPPUNIT_ASSERT(eventEntries[6].event() == 7);

  std::vector<EventNumber_t>().swap(eventNumbers);
  std::vector<IndexIntoFile::EventEntry>().swap(eventEntries);

  CPPUNIT_ASSERT(indexIntoFile.numberOfEvents() == 7);

  TestEventFinder* ptr(new TestEventFinder);
  ptr->push_back(7);
  ptr->push_back(6);
  ptr->push_back(10);
  ptr->push_back(9);
  ptr->push_back(5);
  ptr->push_back(4);
  ptr->push_back(8);

  boost::shared_ptr<IndexIntoFile::EventFinder> shptr(ptr);
  indexIntoFile.setEventFinder(shptr);

  indexIntoFile.fillEventNumbers();
  CPPUNIT_ASSERT(eventNumbers[0] == 9);
  CPPUNIT_ASSERT(eventNumbers[1] == 10);
  CPPUNIT_ASSERT(eventNumbers[2] == 8);
  CPPUNIT_ASSERT(eventNumbers[3] == 4);
  CPPUNIT_ASSERT(eventNumbers[4] == 5);
  CPPUNIT_ASSERT(eventNumbers[5] == 6);
  CPPUNIT_ASSERT(eventNumbers[6] == 7);

  indexIntoFile.fillEventEntries();
  CPPUNIT_ASSERT(eventEntries[0].event() == 9);
  CPPUNIT_ASSERT(eventEntries[1].event() == 10);
  CPPUNIT_ASSERT(eventEntries[2].event() == 8);
  CPPUNIT_ASSERT(eventEntries[3].event() == 4);
  CPPUNIT_ASSERT(eventEntries[4].event() == 5);
  CPPUNIT_ASSERT(eventEntries[5].event() == 6);
  CPPUNIT_ASSERT(eventEntries[6].event() == 7);

  std::vector<EventNumber_t>().swap(eventNumbers);
  std::vector<IndexIntoFile::EventEntry>().swap(eventEntries);

  indexIntoFile.fillEventEntries();
  CPPUNIT_ASSERT(eventEntries[0].event() == 9);
  CPPUNIT_ASSERT(eventEntries[1].event() == 10);
  CPPUNIT_ASSERT(eventEntries[2].event() == 8);
  CPPUNIT_ASSERT(eventEntries[3].event() == 4);
  CPPUNIT_ASSERT(eventEntries[4].event() == 5);
  CPPUNIT_ASSERT(eventEntries[5].event() == 6);
  CPPUNIT_ASSERT(eventEntries[6].event() == 7);

  indexIntoFile.fillEventNumbers();
  CPPUNIT_ASSERT(eventNumbers[0] == 9);
  CPPUNIT_ASSERT(eventNumbers[1] == 10);
  CPPUNIT_ASSERT(eventNumbers[2] == 8);
  CPPUNIT_ASSERT(eventNumbers[3] == 4);
  CPPUNIT_ASSERT(eventNumbers[4] == 5);
  CPPUNIT_ASSERT(eventNumbers[5] == 6);
  CPPUNIT_ASSERT(eventNumbers[6] == 7);

  std::vector<EventNumber_t>().swap(eventNumbers);
  std::vector<IndexIntoFile::EventEntry>().swap(eventEntries);

  indexIntoFile.fillEventNumbersOrEntries(true, true);
  indexIntoFile.fillEventNumbersOrEntries(true, true);

  CPPUNIT_ASSERT(eventNumbers[0] == 9);
  CPPUNIT_ASSERT(eventNumbers[1] == 10);
  CPPUNIT_ASSERT(eventNumbers[2] == 8);
  CPPUNIT_ASSERT(eventNumbers[3] == 4);
  CPPUNIT_ASSERT(eventNumbers[4] == 5);
  CPPUNIT_ASSERT(eventNumbers[5] == 6);
  CPPUNIT_ASSERT(eventNumbers[6] == 7);

  CPPUNIT_ASSERT(eventEntries[0].event() == 9);
  CPPUNIT_ASSERT(eventEntries[1].event() == 10);
  CPPUNIT_ASSERT(eventEntries[2].event() == 8);
  CPPUNIT_ASSERT(eventEntries[3].event() == 4);
  CPPUNIT_ASSERT(eventEntries[4].event() == 5);
  CPPUNIT_ASSERT(eventEntries[5].event() == 6);
  CPPUNIT_ASSERT(eventEntries[6].event() == 7);

  std::vector<EventNumber_t>().swap(eventNumbers);
  std::vector<IndexIntoFile::EventEntry>().swap(eventEntries);

  std::vector<EventNumber_t>& unsortedEventNumbers = indexIntoFile.unsortedEventNumbers();
  CPPUNIT_ASSERT(!unsortedEventNumbers.empty());
  indexIntoFile.doneFileInitialization();
  CPPUNIT_ASSERT(unsortedEventNumbers.empty());
  CPPUNIT_ASSERT(unsortedEventNumbers.capacity() == 0);
  unsortedEventNumbers.push_back(7);
  unsortedEventNumbers.push_back(6);
  unsortedEventNumbers.push_back(10);
  unsortedEventNumbers.push_back(9);
  unsortedEventNumbers.push_back(5);
  unsortedEventNumbers.push_back(4);
  unsortedEventNumbers.push_back(8);

  indexIntoFile.fillEventNumbersOrEntries(true, true);

  CPPUNIT_ASSERT(eventNumbers[0] == 9);
  CPPUNIT_ASSERT(eventNumbers[1] == 10);
  CPPUNIT_ASSERT(eventNumbers[2] == 8);
  CPPUNIT_ASSERT(eventNumbers[3] == 4);
  CPPUNIT_ASSERT(eventNumbers[4] == 5);
  CPPUNIT_ASSERT(eventNumbers[5] == 6);
  CPPUNIT_ASSERT(eventNumbers[6] == 7);

  CPPUNIT_ASSERT(eventEntries[0].event() == 9);
  CPPUNIT_ASSERT(eventEntries[1].event() == 10);
  CPPUNIT_ASSERT(eventEntries[2].event() == 8);
  CPPUNIT_ASSERT(eventEntries[3].event() == 4);
  CPPUNIT_ASSERT(eventEntries[4].event() == 5);
  CPPUNIT_ASSERT(eventEntries[5].event() == 6);
  CPPUNIT_ASSERT(eventEntries[6].event() == 7);

  indexIntoFile.inputFileClosed();
  CPPUNIT_ASSERT(unsortedEventNumbers.empty());
  CPPUNIT_ASSERT(unsortedEventNumbers.capacity() == 0);
  CPPUNIT_ASSERT(eventEntries.capacity() == 0);
  CPPUNIT_ASSERT(eventEntries.empty());
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiIndexes().capacity() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiIndexes().empty());
  CPPUNIT_ASSERT(indexIntoFile.transient_.eventFinder_.get() == 0);
}

void TestIndexIntoFile::testEmptyIndex() {
  edm::IndexIntoFile indexIntoFile;

  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  CPPUNIT_ASSERT(iterNumEnd.indexIntoFile() == &indexIntoFile);
  CPPUNIT_ASSERT(iterNumEnd.size() == 0);
  CPPUNIT_ASSERT(iterNumEnd.type() == kEnd);
  CPPUNIT_ASSERT(iterNumEnd.indexToRun() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterNumEnd.indexToLumi() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterNumEnd.indexToEventRange() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterNumEnd.indexToEvent() == 0);
  CPPUNIT_ASSERT(iterNumEnd.nEvents() == 0);

  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  CPPUNIT_ASSERT(iterFirstEnd.indexIntoFile() == &indexIntoFile);
  CPPUNIT_ASSERT(iterFirstEnd.size() == 0);
  CPPUNIT_ASSERT(iterFirstEnd.type() == kEnd);
  CPPUNIT_ASSERT(iterFirstEnd.indexToRun() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterFirstEnd.indexToLumi() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterFirstEnd.indexToEventRange() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterFirstEnd.indexToEvent() == 0);
  CPPUNIT_ASSERT(iterFirstEnd.nEvents() == 0);

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  CPPUNIT_ASSERT(iterNum == iterNumEnd);

  skipEventBackward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  CPPUNIT_ASSERT(iterFirst == iterFirstEnd);

  skipEventBackward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);
}

void TestIndexIntoFile::testIterEndWithEvent() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 11, 101, 7, 0); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 6, 1); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 0); // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 1); // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 101, 5, 2); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 4, 3); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 2); // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 102, 5, 4); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 102, 4, 5); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 102, 0, 3); // Lumi
  indexIntoFile.addEntry(fakePHID1, 11,   0, 0, 0); // Run
  indexIntoFile.addEntry(fakePHID2, 11,   0, 0, 1); // Run
  indexIntoFile.addEntry(fakePHID2, 11, 101, 0, 4); // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 5); // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 4, 6); // Event
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 6); // Lumi
  indexIntoFile.addEntry(fakePHID2, 11,   0, 0, 2); // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iter3(&indexIntoFile,
                                             IndexIntoFile::firstAppearanceOrder,
                                             IndexIntoFile::kEvent,
                                             0,
                                             3,
                                             2,
                                             1,
                                             2);
  edm::IndexIntoFile::IndexIntoFileItr iter1(iter3);

  CPPUNIT_ASSERT(iter1 == iter3);
  CPPUNIT_ASSERT(iter1.indexIntoFile() == &indexIntoFile);
  CPPUNIT_ASSERT(iter1.size() == 10);
  CPPUNIT_ASSERT(iter1.type() == kEvent);
  CPPUNIT_ASSERT(iter1.indexToRun() == 0);
  CPPUNIT_ASSERT(iter1.indexToLumi() == 3);
  CPPUNIT_ASSERT(iter1.indexToEventRange() == 2);
  CPPUNIT_ASSERT(iter1.indexToEvent() == 1);
  CPPUNIT_ASSERT(iter1.nEvents() == 2);

  iter1 = ++iter3;
  CPPUNIT_ASSERT(iter1 == iter3);
 

  CPPUNIT_ASSERT(indexIntoFile.iterationWillBeInEntryOrder(IndexIntoFile::firstAppearanceOrder) == true);

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterFirstCopy = iterFirst;
  edm::IndexIntoFile::IndexIntoFileItr iterFirstCopy2 = iterFirst;
  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  int i = 0;
  for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++iterFirstCopy, ++i) {
    CPPUNIT_ASSERT(iterFirst== iterFirstCopy);
    iterFirstCopy2 = iterFirstCopy;
    CPPUNIT_ASSERT(iterFirst== iterFirstCopy2);
    if (i == 0) {
      check(iterFirst, kRun, 0, 1, 1, 0, 2);
      CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterFirst.size() == 10);
    }
    else if (i == 1)  check(iterFirst, kLumi,  0, 1,  1, 0, 2);
    else if (i == 2)  check(iterFirst, kLumi,  0, 2,  1, 0, 2);
    else if (i == 3)  check(iterFirst, kLumi,  0, 3,  1, 0, 2);
    else if (i == 4)  check(iterFirst, kEvent, 0, 3,  1, 0, 2);
    else if (i == 5)  check(iterFirst, kEvent, 0, 3,  1, 1, 2);
    else if (i == 6)  check(iterFirst, kEvent, 0, 3,  3, 0, 2);
    else if (i == 7)  check(iterFirst, kEvent, 0, 3,  3, 1, 2);
    else if (i == 8)  check(iterFirst, kLumi,  0, 4,  4, 0, 2);
    else if (i == 9)  check(iterFirst, kEvent, 0, 4,  4, 0, 2);
    else if (i == 10) check(iterFirst, kEvent, 0, 4,  4, 1, 2);
    else if (i == 11) check(iterFirst, kRun,   5, 7, -1, 0, 0);
    else if (i == 12) check(iterFirst, kRun,   6, 7, -1, 0, 0);
    else if (i == 13) check(iterFirst, kLumi,  6, 7, -1, 0, 0);
    else if (i == 14) check(iterFirst, kLumi,  6, 8,  9, 0, 1);
    else if (i == 15) check(iterFirst, kLumi,  6, 9,  9, 0, 1);
    else if (i == 16) check(iterFirst, kEvent, 6, 9,  9, 0, 1);
    else CPPUNIT_ASSERT(false);

    if (i == 0) CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
    if (i == 10) CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
    if (i == 12) CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 6);

    if (i == 0) CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
    if (i == 10) CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 4);
    if (i == 12) CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
  }
  CPPUNIT_ASSERT(i == 17);

  for (i = 0, iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
       iterFirst != iterFirstEnd;
       ++iterFirst, ++i) {
    if (i == 0) checkIDRunLumiEntry(iterFirst, 0, 11, 0, 0);    // Run
    if (i == 1) checkIDRunLumiEntry(iterFirst, 0, 11, 101, 0);  // Lumi
    if (i == 2) checkIDRunLumiEntry(iterFirst, 0, 11, 101, 1);  // Lumi
    if (i == 3) checkIDRunLumiEntry(iterFirst, 0, 11, 101, 2);  // Lumi
    if (i == 4) checkIDRunLumiEntry(iterFirst, 0, 11, 101, 0);  // Event
    if (i == 5) checkIDRunLumiEntry(iterFirst, 0, 11, 101, 1);  // Event
    if (i == 6) checkIDRunLumiEntry(iterFirst, 0, 11, 101, 2);  // Event
    if (i == 7) checkIDRunLumiEntry(iterFirst, 0, 11, 101, 3);  // Event
    if (i == 8) checkIDRunLumiEntry(iterFirst, 0, 11, 102, 3);  // Lumi
    if (i == 9) checkIDRunLumiEntry(iterFirst, 0, 11, 102, 4);  // Event
    if (i == 10) checkIDRunLumiEntry(iterFirst, 0, 11, 102, 5); // Event
    if (i == 11) checkIDRunLumiEntry(iterFirst, 1, 11, 0, 1);   // Run
    if (i == 12) checkIDRunLumiEntry(iterFirst, 1, 11, 0, 2);   // Run
    if (i == 13) checkIDRunLumiEntry(iterFirst, 1, 11, 101, 4); // Lumi
    if (i == 14) checkIDRunLumiEntry(iterFirst, 1, 11, 102, 5); // Lumi
    if (i == 15) checkIDRunLumiEntry(iterFirst, 1, 11, 102, 6); // Lumi
    if (i == 16) checkIDRunLumiEntry(iterFirst, 1, 11, 102, 6); // Event
  }
  checkIDRunLumiEntry(iterFirst, -1, 0, 0, -1); // Event

  CPPUNIT_ASSERT(indexIntoFile.runOrLumiIndexes().empty());

  // Now repeat the above tests for the sorted iteration

  edm::IndexIntoFile::IndexIntoFileItr iter4(&indexIntoFile,
                                             IndexIntoFile::numericalOrder,
                                             IndexIntoFile::kEvent,
                                             0,
                                             3,
                                             1,
                                             3,
                                             4);
  edm::IndexIntoFile::IndexIntoFileItr iter2(iter4);

  CPPUNIT_ASSERT(iter2 == iter4);
  CPPUNIT_ASSERT(iter2.indexIntoFile() == &indexIntoFile);
  CPPUNIT_ASSERT(iter2.size() == 10);
  CPPUNIT_ASSERT(iter2.type() == kEvent);
  CPPUNIT_ASSERT(iter2.indexToRun() == 0);
  CPPUNIT_ASSERT(iter2.indexToLumi() == 3);
  CPPUNIT_ASSERT(iter2.indexToEventRange() == 1);
  CPPUNIT_ASSERT(iter2.indexToEvent() == 3);
  CPPUNIT_ASSERT(iter2.nEvents() == 4);

  CPPUNIT_ASSERT(!indexIntoFile.runOrLumiIndexes().empty());

  std::vector<IndexIntoFile::EventEntry>&  eventEntries  = indexIntoFile.eventEntries();
  eventEntries.push_back(IndexIntoFile::EventEntry(7, 0));
  eventEntries.push_back(IndexIntoFile::EventEntry(6, 1));
  eventEntries.push_back(IndexIntoFile::EventEntry(5, 2));
  eventEntries.push_back(IndexIntoFile::EventEntry(4, 3));
  eventEntries.push_back(IndexIntoFile::EventEntry(5, 4));
  eventEntries.push_back(IndexIntoFile::EventEntry(4, 5));
  eventEntries.push_back(IndexIntoFile::EventEntry(4, 6));
  indexIntoFile.sortEventEntries();

  CPPUNIT_ASSERT(indexIntoFile.iterationWillBeInEntryOrder(IndexIntoFile::numericalOrder) == false);

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterNumCopy = iterNum;
  edm::IndexIntoFile::IndexIntoFileItr iterNumCopy2 = iterNum;
  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  i = 0;
  for (; iterNum != iterNumEnd; ++iterNum, ++iterNumCopy, ++i) {
    iterNumCopy2 = iterNumCopy;
    CPPUNIT_ASSERT(iterNum == iterNumCopy);
    CPPUNIT_ASSERT(iterNum == iterNumCopy2);
    if (i == 0) {
      check(iterNum, kRun, 0, 1, 1, 0, 4);
      CPPUNIT_ASSERT(iterNum.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterNum.size() == 10);
    }
    else if (i == 1)  check(iterNum, kLumi,  0, 1,  1, 0, 4);
    else if (i == 2)  check(iterNum, kLumi,  0, 2,  1, 0, 4);
    else if (i == 3)  check(iterNum, kLumi,  0, 3,  1, 0, 4);
    else if (i == 4)  check(iterNum, kEvent, 0, 3,  1, 0, 4);
    else if (i == 5)  check(iterNum, kEvent, 0, 3,  1, 1, 4);
    else if (i == 6)  check(iterNum, kEvent, 0, 3,  1, 2, 4);
    else if (i == 7)  check(iterNum, kEvent, 0, 3,  1, 3, 4);
    else if (i == 8)  check(iterNum, kLumi,  0, 4,  4, 0, 2);
    else if (i == 9)  check(iterNum, kEvent, 0, 4,  4, 0, 2);
    else if (i == 10) check(iterNum, kEvent, 0, 4,  4, 1, 2);
    else if (i == 11) check(iterNum, kRun,   5, 7, -1, 0, 0);
    else if (i == 12) check(iterNum, kRun,   6, 7, -1, 0, 0);
    else if (i == 13) check(iterNum, kLumi,  6, 7, -1, 0, 0);
    else if (i == 14) check(iterNum, kLumi,  6, 8,  8, 0, 1);
    else if (i == 15) check(iterNum, kLumi,  6, 9,  8, 0, 1);
    else if (i == 16) check(iterNum, kEvent, 6, 9,  8, 0, 1);
    else CPPUNIT_ASSERT(false);

    if (i == 0) CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == 3);
    if (i == 10) CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == 3);
    if (i == 12) CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == 6);

    if (i == 0) CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == 3);
    if (i == 10) CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == 5);
    if (i == 12) CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
  }
  CPPUNIT_ASSERT(i == 17);

  for (i = 0, iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
       iterNum != iterNumEnd;
       ++iterNum, ++i) {
    if (i == 0) checkIDRunLumiEntry(iterNum, 0, 11, 0, 0);    // Run
    if (i == 1) checkIDRunLumiEntry(iterNum, 0, 11, 101, 0);  // Lumi
    if (i == 2) checkIDRunLumiEntry(iterNum, 0, 11, 101, 1);  // Lumi
    if (i == 3) checkIDRunLumiEntry(iterNum, 0, 11, 101, 2);  // Lumi
    if (i == 4) checkIDRunLumiEntry(iterNum, 0, 11, 101, 3);  // Event
    if (i == 5) checkIDRunLumiEntry(iterNum, 0, 11, 101, 2);  // Event
    if (i == 6) checkIDRunLumiEntry(iterNum, 0, 11, 101, 1);  // Event
    if (i == 7) checkIDRunLumiEntry(iterNum, 0, 11, 101, 0);  // Event
    if (i == 8) checkIDRunLumiEntry(iterNum, 0, 11, 102, 3);  // Lumi
    if (i == 9) checkIDRunLumiEntry(iterNum, 0, 11, 102, 5);  // Event
    if (i == 10) checkIDRunLumiEntry(iterNum, 0, 11, 102, 4); // Event
    if (i == 11) checkIDRunLumiEntry(iterNum, 1, 11, 0, 1);   // Run
    if (i == 12) checkIDRunLumiEntry(iterNum, 1, 11, 0, 2);   // Run
    if (i == 13) checkIDRunLumiEntry(iterNum, 1, 11, 101, 4); // Lumi
    if (i == 14) checkIDRunLumiEntry(iterNum, 1, 11, 102, 5); // Lumi
    if (i == 15) checkIDRunLumiEntry(iterNum, 1, 11, 102, 6); // Lumi
    if (i == 16) checkIDRunLumiEntry(iterNum, 1, 11, 102, 6); // Event
  }
  checkIDRunLumiEntry(iterFirst, -1, 0, 0, -1); // Event


  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  check(iterFirst, kRun, 0, 1, 1, 0, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 0);
  check(iterFirst, kRun, 0, 1, 1, 1, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 1);
  check(iterFirst, kRun, 0, 1, 3, 0, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 2);
  check(iterFirst, kRun, 0, 1, 3, 1, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  check(iterFirst, kRun, 0, 4, 4, 0, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  check(iterFirst, kRun, 0, 4, 4, 1, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  check(iterFirst, kRun, 5, 7, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(1, 11, 102, 6);
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 0);
  check(iterFirst, kLumi, 0, 1, 1, 1, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 1);
  check(iterFirst, kLumi, 0, 1, 3, 0, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 2);
  check(iterFirst, kLumi, 0, 1, 3, 1, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  check(iterFirst, kLumi, 0, 4, 4, 0, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  check(iterFirst, kLumi, 0, 4, 4, 1, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  check(iterFirst, kRun, 5, 7, -1, 0, 0);

  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  skipEventForward(iterFirst);
  checkSkipped(1, 11, 102, 6);
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);


  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 0);
  check(iterFirst, kEvent, 0, 3, 1, 1, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 1);
  check(iterFirst, kEvent, 0, 3, 3, 0, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 2);
  check(iterFirst, kEvent, 0, 3, 3, 1, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  check(iterFirst, kLumi, 0, 4, 4, 0, 2);

  ++iterFirst;
  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  check(iterFirst, kEvent, 0, 4, 4, 1, 2);

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  check(iterFirst, kRun, 5, 7, -1, 0, 0);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kLumi, 0, 1, 1, 0 , 2);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kLumi, 0, 4, 4, 0 , 2);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kRun, 5, 7, -1, 0 , 0);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kLumi, 6, 7, -1, 0 , 0);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kLumi, 6, 8, 9, 0 , 1);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  check(iterFirst, kEvent, 0, 3, 1, 0, 2);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kLumi, 0, 4, 4, 0, 2);
  ++iterFirst;
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kRun, 5, 7, -1, 0, 0);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  check(iterFirst, kEvent, 0, 3, 1, 0, 2);
  iterFirst.advanceToNextRun();
  check(iterFirst, kRun, 5, 7, -1, 0, 0);

  // Repeat skip tests with the other sort order

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  check(iterNum, kRun, 0, 1, 1, 0, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 3);
  check(iterNum, kRun, 0, 1, 1, 1, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 2);
  check(iterNum, kRun, 0, 1, 1, 2, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 1);
  check(iterNum, kRun, 0, 1, 1, 3, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 0);
  check(iterNum, kRun, 0, 4, 4, 0, 2);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 5);
  check(iterNum, kRun, 0, 4, 4, 1, 2);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 4);
  check(iterNum, kRun, 5, 7, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(1, 11, 102, 6);
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 3);
  check(iterNum, kLumi, 0, 1, 1, 1, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 2);
  check(iterNum, kLumi, 0, 1, 1, 2, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 1);
  check(iterNum, kLumi, 0, 1, 1, 3, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 0);
  check(iterNum, kLumi, 0, 4, 4, 0, 2);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 5);
  check(iterNum, kLumi, 0, 4, 4, 1, 2);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 4);
  check(iterNum, kRun, 5, 7, -1, 0, 0);

  ++iterNum;
  ++iterNum;
  ++iterNum;
  check(iterNum, kLumi, 6, 8, 8, 0, 1);
  skipEventForward(iterNum);
  checkSkipped(1, 11, 102, 6);
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 3);
  check(iterNum, kEvent, 0, 3, 1, 1, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 2);
  check(iterNum, kEvent, 0, 3, 1, 2, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 1);
  check(iterNum, kEvent, 0, 3, 1, 3, 4);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 0);
  check(iterNum, kLumi, 0, 4, 4, 0, 2);

  ++iterNum;
  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 5);
  check(iterNum, kEvent, 0, 4, 4, 1, 2);

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 4);
  check(iterNum, kRun, 5, 7, -1, 0, 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kLumi, 0, 1, 1, 0 , 4);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kLumi, 0, 4, 4, 0 , 2);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kRun, 5, 7, -1, 0 , 0);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kLumi, 6, 7, -1, 0 , 0);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kLumi, 6, 8, 8, 0 , 1);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  check(iterNum, kEvent, 0, 3, 1, 0 , 4);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kLumi, 0, 4, 4, 0, 2);
  ++iterNum;
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kRun, 5, 7, -1, 0, 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  check(iterNum, kEvent, 0, 3, 1, 0, 4);
  iterNum.advanceToNextRun();
  check(iterNum, kRun, 5, 7, -1, 0, 0);

  // Check backwards iteration

  iterFirst = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);

  skipEventBackward(iterFirst);
  checkSkipped(1, 11, 102, 6);
  check(iterFirst, kRun, 5, 8, 9, 0 , 1);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  check(iterFirst, kRun, 0, 4, 4, 1, 2);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  check(iterFirst, kRun, 0, 4, 4, 0, 2);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  check(iterFirst, kRun, 0, 1, 3, 1, 2);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 2);
  check(iterFirst, kRun, 0, 1, 3, 0, 2);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 1);
  check(iterFirst, kRun, 0, 1, 1, 1, 2);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 0);
  check(iterFirst, kRun, 0, 1, 1, 0, 2);

  skipEventBackward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kRun, 0, 1, 1, 0, 2);

  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  check(iterFirst, kEvent, 0, 4, 4, 1, 2);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  check(iterFirst, kEvent, 0, 4, 4, 0, 2);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  check(iterFirst, kLumi, 0, 1, 3, 1, 2);

  skipEventForward(iterFirst);
  skipEventForward(iterFirst);
  check(iterFirst, kLumi, 0, 4, 4, 1, 2);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  check(iterFirst, kLumi, 0, 4, 4, 0, 2);

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  check(iterFirst, kLumi, 0, 1, 3, 1, 2);

  iterFirst.advanceToNextRun();
  iterFirst.advanceToEvent();
  check(iterFirst, kEvent, 6, 9, 9, 0, 1);
  
  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  check(iterFirst, kRun, 0, 4, 4, 1, 2);

  iterFirst.advanceToNextRun();
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  check(iterFirst, kLumi, 6, 8, 9, 0, 1);
  
  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  check(iterFirst, kRun, 0, 4, 4, 1, 2);

  iterNum = indexIntoFile.end(IndexIntoFile::numericalOrder);

  skipEventBackward(iterNum);
  checkSkipped(1, 11, 102, 6);
  check(iterNum, kRun, 5, 8, 8, 0 , 1);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 4);
  check(iterNum, kRun, 0, 4, 4, 1, 2);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 5);
  check(iterNum, kRun, 0, 4, 4, 0, 2);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 0);
  check(iterNum, kRun, 0, 1, 1, 3, 4);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 1);
  check(iterNum, kRun, 0, 1, 1, 2, 4);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 2);
  check(iterNum, kRun, 0, 1, 1, 1, 4);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 3);
  check(iterNum, kRun, 0, 1, 1, 0, 4);

  skipEventBackward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kRun, 0, 1, 1, 0, 4);

  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  check(iterNum, kEvent, 0, 4, 4, 1, 2);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 5);
  check(iterNum, kEvent, 0, 4, 4, 0, 2);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 0);
  check(iterNum, kLumi, 0, 1, 1, 3, 4);

  skipEventForward(iterNum);
  skipEventForward(iterNum);
  check(iterNum, kLumi, 0, 4, 4, 1, 2);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 5);
  check(iterNum, kLumi, 0, 4, 4, 0, 2);

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 0);
  check(iterNum, kLumi, 0, 1, 1, 3, 4);

  iterNum.advanceToNextRun();
  iterNum.advanceToEvent();
  check(iterNum, kEvent, 6, 9, 8, 0, 1);
  
  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 4);
  check(iterNum, kRun, 0, 4, 4, 1, 2);

  iterNum.advanceToNextRun();
  ++iterNum;
  ++iterNum;
  ++iterNum;
  check(iterNum, kLumi, 6, 8, 8, 0, 1);
  
  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 4);
  check(iterNum, kRun, 0, 4, 4, 1, 2);
}


void TestIndexIntoFile::testIterEndWithLumi() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 0); // Lumi
  indexIntoFile.addEntry(fakePHID1, 11,   0, 0, 0); // Run
  indexIntoFile.addEntry(fakePHID1, 12, 101, 0, 1); // Lumi
  indexIntoFile.addEntry(fakePHID1, 12, 101, 0, 2); // Lumi
  indexIntoFile.addEntry(fakePHID1, 12,   0, 0, 1); // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  int i = 0;
  for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
    if (i == 0)       check(iterFirst, kRun,   0, 1, -1, 0, 0);
    else if (i == 1)  check(iterFirst, kLumi,  0, 1, -1, 0, 0);
    else if (i == 2)  check(iterFirst, kRun,   2, 3, -1, 0, 0);
    else if (i == 3)  check(iterFirst, kLumi,  2, 3, -1, 0, 0);
    else if (i == 4)  check(iterFirst, kLumi,  2, 4, -1, 0, 0);
    else CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 5);

  // Now repeat the above tests for the sorted iteration

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  for (i = 0; iterNum != iterNumEnd; ++iterNum, ++i) {
    if (i == 0)       check(iterNum, kRun,   0, 1, -1, 0, 0);
    else if (i == 1)  check(iterNum, kLumi,  0, 1, -1, 0, 0);
    else if (i == 2)  check(iterNum, kRun,   2, 3, -1, 0, 0);
    else if (i == 3)  check(iterNum, kLumi,  2, 3, -1, 0, 0);
    else if (i == 4)  check(iterNum, kLumi,  2, 4, -1, 0, 0);
    else CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 5);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0 , 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0 , 0);
}

void TestIndexIntoFile::testIterEndWithRun() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1,   0, 0, 0); // Run
  indexIntoFile.addEntry(fakePHID1, 1,   0, 0, 1); // Run
  indexIntoFile.addEntry(fakePHID1, 2,   0, 0, 2); // Run
  indexIntoFile.addEntry(fakePHID1, 3,   0, 0, 3); // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  int i = 0;
  for (; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
    if (i == 0)       check(iterFirst, kRun,  0, -1, -1, 0, 0);
    else if (i == 1)  check(iterFirst, kRun,  1, -1, -1, 0, 0);
    else if (i == 2)  check(iterFirst, kRun,  2, -1, -1, 0, 0);
    else if (i == 3)  check(iterFirst, kRun,  3, -1, -1, 0, 0);
    else CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 4);

  // Now repeat the above tests for the sorted iteration

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  i = 0;
  for (; iterNum != iterNumEnd; ++iterNum, ++i) {
    if (i == 0)       check(iterNum, kRun,   0, -1, -1, 0, 0);
    else if (i == 1)  check(iterNum, kRun,   1, -1, -1, 0, 0);
    else if (i == 2)  check(iterNum, kRun,   2, -1, -1, 0, 0);
    else if (i == 3)  check(iterNum, kRun,   3, -1, -1, 0, 0);
    else CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 4);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0 , 0);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kRun, 2, -1, -1, 0 , 0);
  ++iterFirst;
  check(iterFirst, kRun, 3, -1, -1, 0 , 0);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kEnd, -1, -1, -1, 0 , 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0 , 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kRun, 2, -1, -1, 0 , 0);
  ++iterNum;
  check(iterNum, kRun, 3, -1, -1, 0 , 0);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kEnd, -1, -1, -1, 0 , 0);
}

void TestIndexIntoFile::testIterLastLumiRangeNoEvents() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 101, 5, 0); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 101, 0, 0); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 101, 0, 1); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 102, 6, 1); // Event
  indexIntoFile.addEntry(fakePHID1, 1, 102, 0, 2); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 102, 0, 3); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1,   0, 0, 0); // Run
  indexIntoFile.addEntry(fakePHID1, 2, 101, 7, 2); // Event
  indexIntoFile.addEntry(fakePHID1, 2, 101, 0, 4); // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 101, 0, 5); // Lumi
  indexIntoFile.addEntry(fakePHID1, 2,   0, 0, 1); // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  int i = 0;
  for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
    if (i == 0)       check(iterFirst, kRun,   0, 1, 1, 0, 1);
    else if (i == 1)  check(iterFirst, kLumi,  0, 1, 1, 0, 1);
    else if (i == 2)  check(iterFirst, kLumi,  0, 2, 1, 0, 1);
    else if (i == 3)  check(iterFirst, kEvent, 0, 2, 1, 0, 1);
    else if (i == 4)  check(iterFirst, kLumi,  0, 3, 3, 0, 1);
    else if (i == 5)  check(iterFirst, kLumi,  0, 4, 3, 0, 1);
    else if (i == 6)  check(iterFirst, kEvent, 0, 4, 3, 0, 1);
    else if (i == 7)  check(iterFirst, kRun,   5, 6, 6, 0, 1);
    else if (i == 8)  check(iterFirst, kLumi,  5, 6, 6, 0, 1);
    else if (i == 9)  check(iterFirst, kLumi,  5, 7, 6, 0, 1);
    else if (i == 10) check(iterFirst, kEvent, 5, 7, 6, 0, 1);
    else CPPUNIT_ASSERT(false);
  }
  CPPUNIT_ASSERT(i == 11);

  std::vector<IndexIntoFile::EventEntry>&  eventEntries  = indexIntoFile.eventEntries();
  eventEntries.push_back(IndexIntoFile::EventEntry(5, 0));
  eventEntries.push_back(IndexIntoFile::EventEntry(6, 1));
  eventEntries.push_back(IndexIntoFile::EventEntry(7, 2));
  indexIntoFile.sortEventEntries();

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  i = 0;
  for (; iterNum != iterNumEnd; ++iterNum, ++i) {
    if (i == 0)       check(iterNum, kRun,   0, 1, 1, 0, 1);
    else if (i == 1)  check(iterNum, kLumi,  0, 1, 1, 0, 1);
    else if (i == 2)  check(iterNum, kLumi,  0, 2, 1, 0, 1);
    else if (i == 3)  check(iterNum, kEvent, 0, 2, 1, 0, 1);
    else if (i == 4)  check(iterNum, kLumi,  0, 3, 3, 0, 1);
    else if (i == 5)  check(iterNum, kLumi,  0, 4, 3, 0, 1);
    else if (i == 6)  check(iterNum, kEvent, 0, 4, 3, 0, 1);
    else if (i == 7)  check(iterNum, kRun,   5, 6, 6, 0, 1);
    else if (i == 8)  check(iterNum, kLumi,  5, 6, 6, 0, 1);
    else if (i == 9)  check(iterNum, kLumi,  5, 7, 6, 0, 1);
    else if (i == 10) check(iterNum, kEvent, 5, 7, 6, 0, 1);
    else CPPUNIT_ASSERT(false);
  }
  CPPUNIT_ASSERT(i == 11);
}

void TestIndexIntoFile::testSkip() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1,   101, 1001, 0); // Event
  indexIntoFile.addEntry(fakePHID1, 1,   101,    0, 0); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1,   101,    0, 1); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1,   102,    0, 2); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1,     0,    0, 0); // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kRun, 0, 3, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0 , 0);

  skipEventBackward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kRun, 0, 1, 1, 0 , 1);

  skipEventBackward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kRun, 0, 1, 1, 0 , 1);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  skipEventForward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kLumi, 0, 3, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0 , 0);

  std::vector<IndexIntoFile::EventEntry>&  eventEntries  = indexIntoFile.eventEntries();
  eventEntries.push_back(IndexIntoFile::EventEntry(1001, 0));
  indexIntoFile.sortEventEntries();

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kRun, 0, 3, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0 , 0);

  skipEventBackward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kRun, 0, 1, 1, 0 , 1);

  skipEventBackward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kRun, 0, 1, 1, 0 , 1);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  skipEventForward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kLumi, 0, 3, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0 , 0);
}

void TestIndexIntoFile::testSkip2() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1,   101, 1001, 0); // Event
  indexIntoFile.addEntry(fakePHID1, 1,   101,    0, 0); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1,   101,    0, 1); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1,   102,    0, 2); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1,     0,    0, 0); // Run
  indexIntoFile.addEntry(fakePHID1, 2,   101, 1001, 1); // Event
  indexIntoFile.addEntry(fakePHID1, 2,   101,    0, 3); // Lumi
  indexIntoFile.addEntry(fakePHID1, 2,   101,    0, 4); // Lumi
  indexIntoFile.addEntry(fakePHID1, 2,   102,    0, 5); // Lumi
  indexIntoFile.addEntry(fakePHID1, 2,     0,    0, 1); // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kRun, 0, 3, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(0, 2, 101, 1);
  check(iterFirst, kRun, 4, 7, -1, 0 , 0);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0 , 0);

  skipEventBackward(iterFirst);
  checkSkipped(0, 2, 101, 1);
  check(iterFirst, kRun, 4, 5, 5, 0 , 1);

  skipEventBackward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kRun, 0, 1, 1, 0 , 1);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  skipEventForward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kLumi, 0, 3, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(0, 2, 101, 1);
  check(iterFirst, kRun, 4, 7, -1, 0 , 0);

  std::vector<IndexIntoFile::EventEntry>&  eventEntries  = indexIntoFile.eventEntries();
  eventEntries.push_back(IndexIntoFile::EventEntry(1001, 0));
  eventEntries.push_back(IndexIntoFile::EventEntry(1001, 1));
  indexIntoFile.sortEventEntries();

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kRun, 0, 3, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(0, 2, 101, 1);
  check(iterNum, kRun, 4, 7, -1, 0 , 0);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0 , 0);

  skipEventBackward(iterNum);
  checkSkipped(0, 2, 101, 1);
  check(iterNum, kRun, 4, 5, 5, 0 , 1);

  skipEventBackward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kRun, 0, 1, 1, 0 , 1);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  skipEventForward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kLumi, 0, 3, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(0, 2, 101, 1);
  check(iterNum, kRun, 4, 7, -1, 0 , 0);
}

void TestIndexIntoFile::testSkip3() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1,     1,    0, 0); // Lumi
  indexIntoFile.addEntry(fakePHID1, 1,     0,    0, 0); // Run
  indexIntoFile.addEntry(fakePHID1, 2,   101,    0, 1); // Lumi
  indexIntoFile.addEntry(fakePHID1, 2,   101,    0, 2); // Lumi
  indexIntoFile.addEntry(fakePHID1, 2,   102, 1001, 0); // Event
  indexIntoFile.addEntry(fakePHID1, 2,   102,    0, 3); // Lumi
  indexIntoFile.addEntry(fakePHID1, 2,   103,    0, 4); // Lumi
  indexIntoFile.addEntry(fakePHID1, 2,     0,    0, 1); // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(0, 2, 102, 0);
  check(iterFirst, kRun, 2, 6, -1, 0, 0);

  std::vector<IndexIntoFile::EventEntry>&  eventEntries  = indexIntoFile.eventEntries();
  eventEntries.push_back(IndexIntoFile::EventEntry(1001, 0));
  indexIntoFile.sortEventEntries();

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(0, 2, 102, 0);
  check(iterNum, kRun, 2, 6, -1, 0, 0);
}

void TestIndexIntoFile::testFind() {

  for (int j = 0; j < 2; ++j) {

    edm::IndexIntoFile indexIntoFile;
    indexIntoFile.addEntry(fakePHID1, 1,   0, 0, 0); // Run
    indexIntoFile.addEntry(fakePHID1, 2,  12, 7, 0); // Event
    indexIntoFile.addEntry(fakePHID1, 2,  12, 6, 1); // Event
    indexIntoFile.addEntry(fakePHID1, 2,  12, 0, 0); // Lumi
    indexIntoFile.addEntry(fakePHID1, 2,  12, 5, 2); // Event
    indexIntoFile.addEntry(fakePHID1, 2,  12, 4, 3); // Event
    indexIntoFile.addEntry(fakePHID1, 2,  12, 0, 1); // Lumi
    indexIntoFile.addEntry(fakePHID1, 2,   0, 0, 1); // Run
    indexIntoFile.addEntry(fakePHID2, 3,   0, 0, 2); // Run
    indexIntoFile.addEntry(fakePHID2, 4,  12, 7, 4); // Event
    indexIntoFile.addEntry(fakePHID2, 4,  12, 6, 5); // Event
    indexIntoFile.addEntry(fakePHID2, 4,  12, 0, 2); // Lumi
    indexIntoFile.addEntry(fakePHID2, 4,  12, 5, 6); // Event
    indexIntoFile.addEntry(fakePHID2, 4,  12, 4, 7); // Event
    indexIntoFile.addEntry(fakePHID2, 4,  12, 0, 3); // Lumi
    indexIntoFile.addEntry(fakePHID2, 4,   0, 0, 3); // Run
    indexIntoFile.addEntry(fakePHID3, 5,   0, 0, 4); // Run
    indexIntoFile.addEntry(fakePHID3, 6,  12, 7, 8); // Event
    indexIntoFile.addEntry(fakePHID3, 6,  12, 6, 9); // Event
    indexIntoFile.addEntry(fakePHID3, 6,  12, 0, 4); // Lumi
    indexIntoFile.addEntry(fakePHID3, 6,  100, 0, 5); // Lumi
    indexIntoFile.addEntry(fakePHID3, 6,   0, 0, 5); // Run
    indexIntoFile.sortVector_Run_Or_Lumi_Entries();

    if (j == 0) {
      std::vector<IndexIntoFile::EventEntry>&  eventEntries  = indexIntoFile.eventEntries();
      eventEntries.push_back(IndexIntoFile::EventEntry(7, 0));
      eventEntries.push_back(IndexIntoFile::EventEntry(6, 1));
      eventEntries.push_back(IndexIntoFile::EventEntry(5, 2));
      eventEntries.push_back(IndexIntoFile::EventEntry(4, 3));
      eventEntries.push_back(IndexIntoFile::EventEntry(7, 4));
      eventEntries.push_back(IndexIntoFile::EventEntry(6, 5));
      eventEntries.push_back(IndexIntoFile::EventEntry(5, 6));
      eventEntries.push_back(IndexIntoFile::EventEntry(4, 7));
      eventEntries.push_back(IndexIntoFile::EventEntry(7, 8));
      eventEntries.push_back(IndexIntoFile::EventEntry(6, 9));
      indexIntoFile.sortEventEntries(); 
    }
    else if (j == 1) {
      std::vector<EventNumber_t>&  eventNumbers  = indexIntoFile.eventNumbers();
      eventNumbers.push_back(7);
      eventNumbers.push_back(6);
      eventNumbers.push_back(5);
      eventNumbers.push_back(4);
      eventNumbers.push_back(7);
      eventNumbers.push_back(6);
      eventNumbers.push_back(5);
      eventNumbers.push_back(4);
      eventNumbers.push_back(7);
      eventNumbers.push_back(6);
      indexIntoFile.sortEvents(); 
    }

    TestEventFinder* ptr(new TestEventFinder);
    ptr->push_back(7);
    ptr->push_back(6);
    ptr->push_back(5);
    ptr->push_back(4);
    ptr->push_back(7);
    ptr->push_back(6);
    ptr->push_back(5);
    ptr->push_back(4);
    ptr->push_back(7);
    ptr->push_back(6);
    boost::shared_ptr<IndexIntoFile::EventFinder> shptr(ptr);
    indexIntoFile.setEventFinder(shptr);

    edm::IndexIntoFile::IndexIntoFileItr iter = indexIntoFile.findPosition(1000, 0, 0);
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));  

    iter = indexIntoFile.findPosition(2, 1, 0);
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));  

    iter = indexIntoFile.findPosition(2, 100, 0);
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));  

    iter = indexIntoFile.findPosition(6, 100, 1);
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));  

    iter = indexIntoFile.findPosition(2, 12, 3);
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));  

    iter = indexIntoFile.findPosition(2, 12, 8);
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));  

    iter = indexIntoFile.findPosition(1, 0, 1);
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));  

    iter = indexIntoFile.findPosition(6, 0, 100);
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));  

    iter = indexIntoFile.findPosition(6, 100, 0);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 9);
    CPPUNIT_ASSERT(iter.indexToLumi() == 11);
    CPPUNIT_ASSERT(iter.indexToEventRange() == IndexIntoFile::invalidIndex);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 0);

    iter = indexIntoFile.findPosition(1, 0, 0);
    edm::IndexIntoFile::IndexIntoFileItr iter1 = indexIntoFile.findPosition(1, 1, 0);
    edm::IndexIntoFile::IndexIntoFileItr iter2 = indexIntoFile.findPosition(1, 1, 1);
    CPPUNIT_ASSERT(iter1 == indexIntoFile.end(IndexIntoFile::numericalOrder));  
    CPPUNIT_ASSERT(iter2 == indexIntoFile.end(IndexIntoFile::numericalOrder));  
    CPPUNIT_ASSERT(iter.indexIntoFile() == &indexIntoFile);
    CPPUNIT_ASSERT(iter.size() == 12);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 0);
    CPPUNIT_ASSERT(iter.indexToLumi() == IndexIntoFile::invalidIndex);
    CPPUNIT_ASSERT(iter.indexToEventRange() == IndexIntoFile::invalidIndex);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 0);

    iter = indexIntoFile.findPosition(2, 0, 0);
    iter1 = indexIntoFile.findPosition(2, 12, 0);
    iter2 = indexIntoFile.findPosition(2, 12, 4);
    edm::IndexIntoFile::IndexIntoFileItr iter3 = indexIntoFile.findPosition(2, 0, 4);
    edm::IndexIntoFile::IndexIntoFileItr iter4 = indexIntoFile.findRunPosition(2);
    CPPUNIT_ASSERT(indexIntoFile.containsItem(2, 0, 0));
    CPPUNIT_ASSERT(!indexIntoFile.containsItem(2000, 0, 0));
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter == iter2);
    CPPUNIT_ASSERT(iter == iter3);
    CPPUNIT_ASSERT(iter == iter4);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(3, 0, 0);
    iter1 = indexIntoFile.findPosition(3, 1, 0);
    iter2 = indexIntoFile.findPosition(3, 1, 1);
    CPPUNIT_ASSERT(iter1 == indexIntoFile.end(IndexIntoFile::numericalOrder));  
    CPPUNIT_ASSERT(iter2 == indexIntoFile.end(IndexIntoFile::numericalOrder));  
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 4);
    CPPUNIT_ASSERT(iter.indexToLumi() == IndexIntoFile::invalidIndex);
    CPPUNIT_ASSERT(iter.indexToEventRange() == IndexIntoFile::invalidIndex);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 0);

    iter = indexIntoFile.findPosition(4, 0, 0);
    iter1 = indexIntoFile.findPosition(4, 12, 0);
    iter2 = indexIntoFile.findPosition(4, 12, 4);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter == iter2);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 5);
    CPPUNIT_ASSERT(iter.indexToLumi() == 6);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 6);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(5, 0, 0);
    iter1 = indexIntoFile.findPosition(5, 1, 0);
    iter2 = indexIntoFile.findPosition(5, 1, 1);
    CPPUNIT_ASSERT(iter1 == indexIntoFile.end(IndexIntoFile::numericalOrder));  
    CPPUNIT_ASSERT(iter2 == indexIntoFile.end(IndexIntoFile::numericalOrder));  
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 8);
    CPPUNIT_ASSERT(iter.indexToLumi() == IndexIntoFile::invalidIndex);
    CPPUNIT_ASSERT(iter.indexToEventRange() == IndexIntoFile::invalidIndex);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 0);

    iter = indexIntoFile.findPosition(6, 0, 0);
    iter1 = indexIntoFile.findPosition(6, 12, 0);
    iter2 = indexIntoFile.findPosition(6, 12, 6);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter == iter2);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 9);
    CPPUNIT_ASSERT(iter.indexToLumi() == 10);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 10);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 2);

    iter = indexIntoFile.findPosition(2, 12, 5);
    iter1 = indexIntoFile.findPosition(2, 0, 5);
    iter2 = indexIntoFile.findPosition(IndexIntoFile::numericalOrder, 2, 0, 5);
    iter3 = indexIntoFile.findPosition(IndexIntoFile::firstAppearanceOrder, 2, 0, 5);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter == iter2);    
    CPPUNIT_ASSERT(iter != iter3);    
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 1);
    CPPUNIT_ASSERT(iter.nEvents() == 4);
    CPPUNIT_ASSERT(iter3.type() == kRun);
    CPPUNIT_ASSERT(iter3.indexToRun() == 1);
    CPPUNIT_ASSERT(iter3.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter3.indexToEventRange() == 3);
    CPPUNIT_ASSERT(iter3.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter3.nEvents() == 2);

    iter3 = indexIntoFile.findPosition(IndexIntoFile::firstAppearanceOrder, 2, 0, 0);
    CPPUNIT_ASSERT(iter3.type() == kRun);
    CPPUNIT_ASSERT(iter3.indexToRun() == 1);
    CPPUNIT_ASSERT(iter3.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter3.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter3.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter3.nEvents() == 2);

    iter3 = indexIntoFile.findPosition(IndexIntoFile::firstAppearanceOrder, 2, 12, 0);
    CPPUNIT_ASSERT(iter3.type() == kRun);
    CPPUNIT_ASSERT(iter3.indexToRun() == 1);
    CPPUNIT_ASSERT(iter3.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter3.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter3.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter3.nEvents() == 2);

    iter3 = indexIntoFile.findPosition(IndexIntoFile::firstAppearanceOrder, 2, 12, 4);
    CPPUNIT_ASSERT(iter3.type() == kRun);
    CPPUNIT_ASSERT(iter3.indexToRun() == 1);
    CPPUNIT_ASSERT(iter3.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter3.indexToEventRange() == 3);
    CPPUNIT_ASSERT(iter3.indexToEvent() == 1);
    CPPUNIT_ASSERT(iter3.nEvents() == 2);

    iter = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
    CPPUNIT_ASSERT(iter != iter3);
    iter.copyPosition(iter3);
    CPPUNIT_ASSERT(iter == iter3);

    iter3 = indexIntoFile.findPosition(IndexIntoFile::firstAppearanceOrder, 6, 100, 0);
    CPPUNIT_ASSERT(iter3.type() == kRun);
    CPPUNIT_ASSERT(iter3.indexToRun() == 9);
    CPPUNIT_ASSERT(iter3.indexToLumi() == 11);
    CPPUNIT_ASSERT(iter3.indexToEventRange() == -1);
    CPPUNIT_ASSERT(iter3.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter3.nEvents() == 0);

    iter = indexIntoFile.findPosition(2, 12, 6);
    iter1 = indexIntoFile.findPosition(2, 0, 6);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 2);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(2, 12, 7);
    iter1 = indexIntoFile.findPosition(2, 0, 7);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 3);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(4, 12, 5);
    iter1 = indexIntoFile.findPosition(4, 0, 5);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 5);
    CPPUNIT_ASSERT(iter.indexToLumi() == 6);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 6);
    CPPUNIT_ASSERT(iter.indexToEvent() == 1);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(4, 12, 6);
    iter1 = indexIntoFile.findPosition(4, 0, 6);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 5);
    CPPUNIT_ASSERT(iter.indexToLumi() == 6);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 6);
    CPPUNIT_ASSERT(iter.indexToEvent() == 2);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(4, 12, 7);
    iter1 = indexIntoFile.findPosition(4, 0, 7);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 5);
    CPPUNIT_ASSERT(iter.indexToLumi() == 6);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 6);
    CPPUNIT_ASSERT(iter.indexToEvent() == 3);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(6, 12, 7);
    iter1 = indexIntoFile.findPosition(6, 0, 7);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 9);
    CPPUNIT_ASSERT(iter.indexToLumi() == 10);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 10);
    CPPUNIT_ASSERT(iter.indexToEvent() == 1);
    CPPUNIT_ASSERT(iter.nEvents() == 2);

    iter = indexIntoFile.findEventPosition(2, 12, 4);
    iter1 = indexIntoFile.findEventPosition(2, 0, 4);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == kEvent);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 3);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findEventPosition(2, 12, 7);
    iter1 = indexIntoFile.findEventPosition(2, 0, 7);
    CPPUNIT_ASSERT(indexIntoFile.containsItem(2, 12, 7));
    CPPUNIT_ASSERT(!indexIntoFile.containsItem(2, 12, 100));
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == kEvent);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 3);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 3);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findEventPosition(2, 12, 100);
    iter1 = indexIntoFile.findEventPosition(2, 0, 100);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));

    iter = indexIntoFile.findLumiPosition(2, 12);
    CPPUNIT_ASSERT(iter.type() == kLumi);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findLumiPosition(2, 100);
    CPPUNIT_ASSERT(indexIntoFile.containsItem(2, 12, 0));
    CPPUNIT_ASSERT(!indexIntoFile.containsItem(2, 100, 0));
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));

    if (j == 0) {
      edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
      iterFirst.advanceToNextRun();
      iterFirst.advanceToNextRun();

      skipEventBackward(iterFirst);
      checkSkipped(0, 2, 12, 3);
      check(iterFirst, kRun, 1, 2, 3, 1, 2);

      edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
      iterNum.advanceToNextRun();
      iterNum.advanceToNextRun();

      skipEventBackward(iterNum);
      checkSkipped(0, 2, 12, 0);
      check(iterNum, kRun, 1, 2, 2, 3, 4);
    }
  }
}

void TestIndexIntoFile::testDuplicateCheckerFunctions() {

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

void TestIndexIntoFile::testReduce() {
  // This test is implemented in FWCore/Integration/test/ProcessHistory_t.cpp
  // because of dependency issues.
}
