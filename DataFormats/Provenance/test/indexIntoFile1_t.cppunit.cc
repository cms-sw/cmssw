/*
 *  indexIntoFile_t.cppunit.cc
 */

#include "cppunit/extensions/HelperMacros.h"

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

class TestIndexIntoFile1 : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestIndexIntoFile1);  
  CPPUNIT_TEST(testRunOrLumiEntry);
  CPPUNIT_TEST(testRunOrLumiIndexes);
  CPPUNIT_TEST(testEventEntry);
  CPPUNIT_TEST(testSortedRunOrLumiItr);
  CPPUNIT_TEST(testKeys);
  CPPUNIT_TEST(testConstructor);
  CPPUNIT_TEST_SUITE_END();
  
public:

  static const IndexIntoFile::EntryType kRun = IndexIntoFile::kRun;
  static const IndexIntoFile::EntryType kLumi = IndexIntoFile::kLumi;
  static const IndexIntoFile::EntryType kEvent = IndexIntoFile::kEvent;
  static const IndexIntoFile::EntryType kEnd = IndexIntoFile::kEnd;

  void setUp() {
    // Make some fake processHistoryID's to work with
    nullPHID = ProcessHistoryID();

    ProcessConfiguration pc;
    std::unique_ptr<ProcessHistory> processHistory1(new ProcessHistory);
    ProcessHistory& ph1 = *processHistory1;
    processHistory1->push_back(pc);
    fakePHID1 = ph1.id();

    std::unique_ptr<ProcessHistory> processHistory2(new ProcessHistory);
    ProcessHistory& ph2 = *processHistory2;
    processHistory2->push_back(pc);
    processHistory2->push_back(pc);
    fakePHID2 = ph2.id();

    std::unique_ptr<ProcessHistory> processHistory3(new ProcessHistory);
    ProcessHistory& ph3 = *processHistory3;
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    fakePHID3 = ph3.id();
  }

  void tearDown() { }

  void testRunOrLumiEntry();
  void testRunOrLumiIndexes();
  void testEventEntry();
  void testSortedRunOrLumiItr();
  void testKeys();
  void testConstructor();

  ProcessHistoryID nullPHID;
  ProcessHistoryID fakePHID1;
  ProcessHistoryID fakePHID2;
  ProcessHistoryID fakePHID3;

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexIntoFile1);

void TestIndexIntoFile1::testRunOrLumiEntry() {

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

void TestIndexIntoFile1::testRunOrLumiIndexes() {

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

void TestIndexIntoFile1::testEventEntry() {
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

void TestIndexIntoFile1::testSortedRunOrLumiItr() {

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

void TestIndexIntoFile1::testKeys() {
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

void TestIndexIntoFile1::testConstructor() {
  edm::IndexIntoFile indexIntoFile;
  CPPUNIT_ASSERT(indexIntoFile.runOrLumiEntries().empty());
  CPPUNIT_ASSERT(indexIntoFile.processHistoryIDs().empty());
  CPPUNIT_ASSERT(indexIntoFile.eventEntries().empty());
  CPPUNIT_ASSERT(indexIntoFile.eventNumbers().empty());
  CPPUNIT_ASSERT(indexIntoFile.setRunOrLumiEntries().empty());
  CPPUNIT_ASSERT(indexIntoFile.setProcessHistoryIDs().empty());
}

