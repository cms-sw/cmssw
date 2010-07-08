/*
 *  indexIntoFile_t.cppunit.cc
 */

#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include <memory>
// This is very ugly, but I am told OK for white box  unit tests 
#define private public
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#undef private

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
  CPPUNIT_TEST(testIterators);
  CPPUNIT_TEST(testFind);
  CPPUNIT_TEST(testDuplicateCheckerFunctions);
  CPPUNIT_TEST_SUITE_END();
  
public:

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
  void testIterators();
  void testFind();
  void testDuplicateCheckerFunctions();

  ProcessHistoryID nullPHID;
  ProcessHistoryID fakePHID1;
  ProcessHistoryID fakePHID2;
  ProcessHistoryID fakePHID3;

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

  eventNumbers.clear();
  eventEntries.clear();

  CPPUNIT_ASSERT(eventEntries[0].event() == 9);
  CPPUNIT_ASSERT(eventEntries[1].event() == 10);
  CPPUNIT_ASSERT(eventEntries[2].event() == 8);
  CPPUNIT_ASSERT(eventEntries[3].event() == 4);
  CPPUNIT_ASSERT(eventEntries[4].event() == 5);
  CPPUNIT_ASSERT(eventEntries[5].event() == 6);
  CPPUNIT_ASSERT(eventEntries[6].event() == 7);

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

  indexIntoFile.inputFileClosed();
  CPPUNIT_ASSERT(eventEntries.capacity() == 0);
  CPPUNIT_ASSERT(eventEntries.empty());
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiIndexes().capacity() == 0);
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiIndexes().empty());
  CPPUNIT_ASSERT(indexIntoFile.transients_.get().eventFinder_.get() == 0);
}

void TestIndexIntoFile::testEmptyIndex() {
  edm::IndexIntoFile indexIntoFile;

  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  CPPUNIT_ASSERT(iterNumEnd.indexIntoFile() == &indexIntoFile);
  CPPUNIT_ASSERT(iterNumEnd.size() == 0);
  CPPUNIT_ASSERT(iterNumEnd.type() == IndexIntoFile::kEnd);
  CPPUNIT_ASSERT(iterNumEnd.indexToRun() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterNumEnd.indexToLumi() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterNumEnd.indexToEventRange() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterNumEnd.indexToEvent() == 0);
  CPPUNIT_ASSERT(iterNumEnd.nEvents() == 0);

  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  CPPUNIT_ASSERT(iterFirstEnd.indexIntoFile() == &indexIntoFile);
  CPPUNIT_ASSERT(iterFirstEnd.size() == 0);
  CPPUNIT_ASSERT(iterFirstEnd.type() == IndexIntoFile::kEnd);
  CPPUNIT_ASSERT(iterFirstEnd.indexToRun() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterFirstEnd.indexToLumi() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterFirstEnd.indexToEventRange() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterFirstEnd.indexToEvent() == 0);
  CPPUNIT_ASSERT(iterFirstEnd.nEvents() == 0);

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  CPPUNIT_ASSERT(iterNum == iterNumEnd);

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  CPPUNIT_ASSERT(iterFirst == iterFirstEnd);
}

void TestIndexIntoFile::testIterators() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 11, 12, 7, 0); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 6, 1); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 0, 0); // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 12, 5, 2); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 4, 3); // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 0, 1); // Lumi
  indexIntoFile.addEntry(fakePHID1, 11,  0, 0, 0); // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  std::vector<IndexIntoFile::EventEntry>&  eventEntries  = indexIntoFile.eventEntries();
  eventEntries.push_back(IndexIntoFile::EventEntry(7, 0));
  eventEntries.push_back(IndexIntoFile::EventEntry(6, 1));
  eventEntries.push_back(IndexIntoFile::EventEntry(5, 2));
  eventEntries.push_back(IndexIntoFile::EventEntry(4, 3));
  indexIntoFile.sortEventEntries();

  CPPUNIT_ASSERT(indexIntoFile.iterationWillBeInEntryOrder(IndexIntoFile::numericalOrder) == false);
  CPPUNIT_ASSERT(indexIntoFile.iterationWillBeInEntryOrder(IndexIntoFile::firstAppearanceOrder) == true);

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterNumCopy = iterNum;
  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  int i = 0;
  for (i = 0; iterNum != iterNumEnd; ++iterNum, ++iterNumCopy, ++i) {
    CPPUNIT_ASSERT(iterNum == iterNumCopy);
    if (i == 0) {
      CPPUNIT_ASSERT(iterNum.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterNum.size() == 3);
      CPPUNIT_ASSERT(iterNum.type() == IndexIntoFile::kRun);
      CPPUNIT_ASSERT(iterNum.indexToRun() == 0);
      CPPUNIT_ASSERT(iterNum.indexToLumi() == 1);
      CPPUNIT_ASSERT(iterNum.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterNum.indexToEvent() == 0);
      CPPUNIT_ASSERT(iterNum.nEvents() == 4);
    }
    if (i == 1) {
      CPPUNIT_ASSERT(iterNum.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterNum.size() == 3);
      CPPUNIT_ASSERT(iterNum.type() == IndexIntoFile::kLumi);
      CPPUNIT_ASSERT(iterNum.indexToRun() == 0);
      CPPUNIT_ASSERT(iterNum.indexToLumi() == 1);
      CPPUNIT_ASSERT(iterNum.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterNum.indexToEvent() == 0);
      CPPUNIT_ASSERT(iterNum.nEvents() == 4);
    }
    if (i == 2) {
      CPPUNIT_ASSERT(iterNum.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterNum.size() == 3);
      CPPUNIT_ASSERT(iterNum.type() == IndexIntoFile::kLumi);
      CPPUNIT_ASSERT(iterNum.indexToRun() == 0);
      CPPUNIT_ASSERT(iterNum.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterNum.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterNum.indexToEvent() == 0);
      CPPUNIT_ASSERT(iterNum.nEvents() == 4);
    }
    if (i == 3) {
      CPPUNIT_ASSERT(iterNum.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterNum.size() == 3);
      CPPUNIT_ASSERT(iterNum.type() == IndexIntoFile::kEvent);
      CPPUNIT_ASSERT(iterNum.indexToRun() == 0);
      CPPUNIT_ASSERT(iterNum.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterNum.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterNum.indexToEvent() == 0);
      CPPUNIT_ASSERT(iterNum.nEvents() == 4);
    }
    if (i == 4) {
      CPPUNIT_ASSERT(iterNum.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterNum.size() == 3);
      CPPUNIT_ASSERT(iterNum.type() == IndexIntoFile::kEvent);
      CPPUNIT_ASSERT(iterNum.indexToRun() == 0);
      CPPUNIT_ASSERT(iterNum.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterNum.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterNum.indexToEvent() == 1);
      CPPUNIT_ASSERT(iterNum.nEvents() == 4);
    }
    if (i == 5) {
      CPPUNIT_ASSERT(iterNum.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterNum.size() == 3);
      CPPUNIT_ASSERT(iterNum.type() == IndexIntoFile::kEvent);
      CPPUNIT_ASSERT(iterNum.indexToRun() == 0);
      CPPUNIT_ASSERT(iterNum.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterNum.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterNum.indexToEvent() == 2);
      CPPUNIT_ASSERT(iterNum.nEvents() == 4);
    }
    if (i == 6) {
      CPPUNIT_ASSERT(iterNum.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterNum.size() == 3);
      CPPUNIT_ASSERT(iterNum.type() == IndexIntoFile::kEvent);
      CPPUNIT_ASSERT(iterNum.indexToRun() == 0);
      CPPUNIT_ASSERT(iterNum.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterNum.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterNum.indexToEvent() == 3);
      CPPUNIT_ASSERT(iterNum.nEvents() == 4);
    }
  }
  CPPUNIT_ASSERT(i == 7);

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterFirstCopy = iterFirst;
  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++iterFirstCopy, ++i) {
    CPPUNIT_ASSERT(iterFirst== iterFirstCopy);
    if (i == 0) {
      CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterFirst.size() == 3);
      CPPUNIT_ASSERT(iterFirst.getEntryType() == IndexIntoFile::kRun);
      CPPUNIT_ASSERT(iterFirst.type() == IndexIntoFile::kRun);
      CPPUNIT_ASSERT(iterFirst.indexToRun() == 0);
      CPPUNIT_ASSERT(iterFirst.indexToLumi() == 1);
      CPPUNIT_ASSERT(iterFirst.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterFirst.indexToEvent() == 0);
      CPPUNIT_ASSERT(iterFirst.nEvents() == 2);
    }
    if (i == 1) {
      CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterFirst.size() == 3);
      CPPUNIT_ASSERT(iterFirst.getEntryType() == IndexIntoFile::kLumi);
      CPPUNIT_ASSERT(iterFirst.type() == IndexIntoFile::kLumi);
      CPPUNIT_ASSERT(iterFirst.indexToRun() == 0);
      CPPUNIT_ASSERT(iterFirst.indexToLumi() == 1);
      CPPUNIT_ASSERT(iterFirst.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterFirst.indexToEvent() == 0);
      CPPUNIT_ASSERT(iterFirst.nEvents() == 2);
    }
    if (i == 2) {
      CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterFirst.size() == 3);
      CPPUNIT_ASSERT(iterFirst.type() == IndexIntoFile::kLumi);
      CPPUNIT_ASSERT(iterFirst.indexToRun() == 0);
      CPPUNIT_ASSERT(iterFirst.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterFirst.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterFirst.indexToEvent() == 0);
      CPPUNIT_ASSERT(iterFirst.nEvents() == 2);
    }
    if (i == 3) {
      CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterFirst.size() == 3);
      CPPUNIT_ASSERT(iterFirst.type() == IndexIntoFile::kEvent);
      CPPUNIT_ASSERT(iterFirst.indexToRun() == 0);
      CPPUNIT_ASSERT(iterFirst.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterFirst.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterFirst.indexToEvent() == 0);
      CPPUNIT_ASSERT(iterFirst.nEvents() == 2);
    }
    if (i == 4) {
      CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterFirst.size() == 3);
      CPPUNIT_ASSERT(iterFirst.type() == IndexIntoFile::kEvent);
      CPPUNIT_ASSERT(iterFirst.indexToRun() == 0);
      CPPUNIT_ASSERT(iterFirst.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterFirst.indexToEventRange() == 1);
      CPPUNIT_ASSERT(iterFirst.indexToEvent() == 1);
      CPPUNIT_ASSERT(iterFirst.nEvents() == 2);
    }
    if (i == 5) {
      CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterFirst.size() == 3);
      CPPUNIT_ASSERT(iterFirst.type() == IndexIntoFile::kEvent);
      CPPUNIT_ASSERT(iterFirst.indexToRun() == 0);
      CPPUNIT_ASSERT(iterFirst.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterFirst.indexToEventRange() == 2);
      CPPUNIT_ASSERT(iterFirst.indexToEvent() == 0);
      CPPUNIT_ASSERT(iterFirst.nEvents() == 2);
    }
    if (i == 6) {
      CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterFirst.size() == 3);
      CPPUNIT_ASSERT(iterFirst.type() == IndexIntoFile::kEvent);
      CPPUNIT_ASSERT(iterFirst.indexToRun() == 0);
      CPPUNIT_ASSERT(iterFirst.indexToLumi() == 2);
      CPPUNIT_ASSERT(iterFirst.indexToEventRange() == 2);
      CPPUNIT_ASSERT(iterFirst.indexToEvent() == 1);
      CPPUNIT_ASSERT(iterFirst.nEvents() == 2);
    }
  }
  CPPUNIT_ASSERT(i == 7);
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
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
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
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
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
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
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
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
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
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
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
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
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
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 9);
    CPPUNIT_ASSERT(iter.indexToLumi() == 10);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 10);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 2);

    iter = indexIntoFile.findPosition(2, 12, 5);
    iter1 = indexIntoFile.findPosition(2, 0, 5);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 1);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(2, 12, 6);
    iter1 = indexIntoFile.findPosition(2, 0, 6);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 2);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(2, 12, 7);
    iter1 = indexIntoFile.findPosition(2, 0, 7);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 3);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(4, 12, 5);
    iter1 = indexIntoFile.findPosition(4, 0, 5);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 5);
    CPPUNIT_ASSERT(iter.indexToLumi() == 6);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 6);
    CPPUNIT_ASSERT(iter.indexToEvent() == 1);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(4, 12, 6);
    iter1 = indexIntoFile.findPosition(4, 0, 6);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 5);
    CPPUNIT_ASSERT(iter.indexToLumi() == 6);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 6);
    CPPUNIT_ASSERT(iter.indexToEvent() == 2);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(4, 12, 7);
    iter1 = indexIntoFile.findPosition(4, 0, 7);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 5);
    CPPUNIT_ASSERT(iter.indexToLumi() == 6);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 6);
    CPPUNIT_ASSERT(iter.indexToEvent() == 3);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findPosition(6, 12, 7);
    iter1 = indexIntoFile.findPosition(6, 0, 7);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kRun);
    CPPUNIT_ASSERT(iter.indexToRun() == 9);
    CPPUNIT_ASSERT(iter.indexToLumi() == 10);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 10);
    CPPUNIT_ASSERT(iter.indexToEvent() == 1);
    CPPUNIT_ASSERT(iter.nEvents() == 2);

    iter = indexIntoFile.findEventPosition(2, 12, 4);
    iter1 = indexIntoFile.findEventPosition(2, 0, 4);
    CPPUNIT_ASSERT(iter == iter1);
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kEvent);
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
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kEvent);
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
    CPPUNIT_ASSERT(iter.type() == IndexIntoFile::kLumi);
    CPPUNIT_ASSERT(iter.indexToRun() == 1);
    CPPUNIT_ASSERT(iter.indexToLumi() == 2);
    CPPUNIT_ASSERT(iter.indexToEventRange() == 2);
    CPPUNIT_ASSERT(iter.indexToEvent() == 0);
    CPPUNIT_ASSERT(iter.nEvents() == 4);

    iter = indexIntoFile.findLumiPosition(2, 100);
    CPPUNIT_ASSERT(indexIntoFile.containsItem(2, 12, 0));
    CPPUNIT_ASSERT(!indexIntoFile.containsItem(2, 100, 0));
    CPPUNIT_ASSERT(iter == indexIntoFile.end(IndexIntoFile::numericalOrder));
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
