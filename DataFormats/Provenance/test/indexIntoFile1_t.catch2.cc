/*
 *  indexIntoFile_t.cppunit.cc
 */

#include <catch2/catch_all.hpp>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"

#include "DataFormats/Provenance/interface/IndexIntoFile.h"

#include <string>
#include <iostream>
#include <memory>

using namespace edm;

class TestIndexIntoFile1 {
public:
  static const IndexIntoFile::EntryType kRun = IndexIntoFile::kRun;
  static const IndexIntoFile::EntryType kLumi = IndexIntoFile::kLumi;
  static const IndexIntoFile::EntryType kEvent = IndexIntoFile::kEvent;
  static const IndexIntoFile::EntryType kEnd = IndexIntoFile::kEnd;

  ProcessHistoryID nullPHID;
  ProcessHistoryID fakePHID1;
  ProcessHistoryID fakePHID2;
  ProcessHistoryID fakePHID3;

  void setUp() {
    // Make some fake processHistoryID's to work with
    nullPHID = ProcessHistoryID();

    ProcessConfiguration pc;
    auto processHistory1 = std::make_unique<ProcessHistory>();
    ProcessHistory& ph1 = *processHistory1;
    processHistory1->push_back(pc);
    fakePHID1 = ph1.id();

    auto processHistory2 = std::make_unique<ProcessHistory>();
    ProcessHistory& ph2 = *processHistory2;
    processHistory2->push_back(pc);
    processHistory2->push_back(pc);
    fakePHID2 = ph2.id();

    auto processHistory3 = std::make_unique<ProcessHistory>();
    ProcessHistory& ph3 = *processHistory3;
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    fakePHID3 = ph3.id();
  }

  void testRunOrLumiEntry() {
    edm::IndexIntoFile::RunOrLumiEntry r1;
    REQUIRE(r1.orderPHIDRun() == edm::IndexIntoFile::invalidEntry);
    REQUIRE(r1.orderPHIDRunLumi() == edm::IndexIntoFile::invalidEntry);
    REQUIRE(r1.entry() == edm::IndexIntoFile::invalidEntry);
    REQUIRE(r1.processHistoryIDIndex() == edm::IndexIntoFile::invalidIndex);
    REQUIRE(r1.run() == edm::IndexIntoFile::invalidRun);
    REQUIRE(r1.lumi() == edm::IndexIntoFile::invalidLumi);
    REQUIRE(r1.beginEvents() == edm::IndexIntoFile::invalidEntry);
    REQUIRE(r1.endEvents() == edm::IndexIntoFile::invalidEntry);

    edm::IndexIntoFile::RunOrLumiEntry r2(1, 2, 3, 4, 5, 6, 7, 8);

    REQUIRE(r2.orderPHIDRun() == 1);
    REQUIRE(r2.orderPHIDRunLumi() == 2);
    REQUIRE(r2.entry() == 3);
    REQUIRE(r2.processHistoryIDIndex() == 4);
    REQUIRE(r2.run() == 5);
    REQUIRE(r2.lumi() == 6);
    REQUIRE(r2.beginEvents() == 7);
    REQUIRE(r2.endEvents() == 8);

    REQUIRE(r2.isRun() == false);

    edm::IndexIntoFile::RunOrLumiEntry r3(1, 2, 3, 4, 5, edm::IndexIntoFile::invalidLumi, 7, 8);

    REQUIRE(r3.isRun() == true);

    r3.setOrderPHIDRun(11);
    REQUIRE(r3.orderPHIDRun() == 11);
    r3.setProcessHistoryIDIndex(12);
    REQUIRE(r3.processHistoryIDIndex() == 12);
    r3.setOrderPHIDRun(1);

    REQUIRE(!(r2 < r3));
    REQUIRE(!(r3 < r2));

    edm::IndexIntoFile::RunOrLumiEntry r4(10, 1, 1, 4, 5, 6, 7, 8);
    REQUIRE(r2 < r4);
    REQUIRE(!(r4 < r2));

    edm::IndexIntoFile::RunOrLumiEntry r5(1, 10, 1, 4, 5, 6, 7, 8);
    REQUIRE(r2 < r5);
    REQUIRE(!(r5 < r2));

    edm::IndexIntoFile::RunOrLumiEntry r6(1, 2, 10, 4, 5, 6, 7, 8);
    REQUIRE(r2 < r6);
    REQUIRE(!(r6 < r2));

    r3.setOrderPHIDRunLumi(1001);
    REQUIRE(r3.orderPHIDRunLumi() == 1001);
  }

  void testRunOrLumiIndexes() {
    edm::IndexIntoFile::RunOrLumiIndexes r1(1, 2, 3, 4);
    REQUIRE(r1.processHistoryIDIndex() == 1);
    REQUIRE(r1.run() == 2);
    REQUIRE(r1.lumi() == 3);
    REQUIRE(r1.indexToGetEntry() == 4);
    REQUIRE(r1.beginEventNumbers() == -1);
    REQUIRE(r1.endEventNumbers() == -1);

    r1.setBeginEventNumbers(11);
    r1.setEndEventNumbers(12);
    REQUIRE(r1.beginEventNumbers() == 11);
    REQUIRE(r1.endEventNumbers() == 12);

    REQUIRE(r1.isRun() == false);

    edm::IndexIntoFile::RunOrLumiIndexes r2(1, 2, edm::IndexIntoFile::invalidLumi, 4);
    REQUIRE(r2.isRun() == true);

    edm::IndexIntoFile::RunOrLumiIndexes r3(1, 2, 3, 4);
    REQUIRE(!(r1 < r3));
    REQUIRE(!(r3 < r1));

    edm::IndexIntoFile::RunOrLumiIndexes r4(11, 2, 3, 4);
    REQUIRE(r1 < r4);
    REQUIRE(!(r4 < r1));

    edm::IndexIntoFile::RunOrLumiIndexes r5(1, 11, 1, 4);
    REQUIRE(r1 < r5);
    REQUIRE(!(r5 < r1));

    edm::IndexIntoFile::RunOrLumiIndexes r6(1, 2, 11, 4);
    REQUIRE(r1 < r6);
    REQUIRE(!(r6 < r1));

    Compare_Index_Run c;
    REQUIRE(!c(r1, r6));
    REQUIRE(!c(r6, r1));
    REQUIRE(c(r1, r5));
    REQUIRE(!c(r5, r1));
    REQUIRE(c(r1, r4));
    REQUIRE(!c(r4, r1));

    Compare_Index c1;
    REQUIRE(!c1(r1, r5));
    REQUIRE(!c1(r5, r1));
    REQUIRE(c1(r1, r4));
    REQUIRE(!c1(r4, r1));
  }

  void testEventEntry() {
    edm::IndexIntoFile::EventEntry e1;
    REQUIRE(e1.event() == edm::IndexIntoFile::invalidEvent);
    REQUIRE(e1.entry() == edm::IndexIntoFile::invalidEntry);

    edm::IndexIntoFile::EventEntry e2(100, 200);
    REQUIRE(e2.event() == 100);
    REQUIRE(e2.entry() == 200);

    edm::IndexIntoFile::EventEntry e3(100, 300);
    edm::IndexIntoFile::EventEntry e4(200, 300);
    edm::IndexIntoFile::EventEntry e5(200, 100);

    REQUIRE(e2 == e3);
    REQUIRE(!(e3 == e4));

    REQUIRE(e3 < e4);
    REQUIRE(e3 < e5);
    REQUIRE(!(e4 < e5));
  }

  void testSortedRunOrLumiItr() {
    edm::IndexIntoFile indexIntoFile0;
    REQUIRE(indexIntoFile0.empty());
    edm::IndexIntoFile::SortedRunOrLumiItr iter(&indexIntoFile0, 0);
    REQUIRE(iter.indexIntoFile() == &indexIntoFile0);
    REQUIRE(iter.runOrLumi() == 0);
    ++iter;
    REQUIRE(iter.runOrLumi() == 0);
    REQUIRE(iter == indexIntoFile0.beginRunOrLumi());
    REQUIRE(iter == indexIntoFile0.endRunOrLumi());

    edm::IndexIntoFile indexIntoFile;
    indexIntoFile.addEntry(fakePHID1, 1, 1, 1, 0);  // Event
    indexIntoFile.addEntry(fakePHID1, 1, 1, 2, 1);  // Event
    indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 0);  // Lumi
    indexIntoFile.addEntry(fakePHID1, 1, 2, 1, 2);  // Event
    indexIntoFile.addEntry(fakePHID1, 1, 2, 2, 3);  // Event
    indexIntoFile.addEntry(fakePHID1, 1, 2, 3, 4);  // Event
    indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 1);  // Lumi
    indexIntoFile.addEntry(fakePHID1, 1, 1, 3, 5);  // Event
    indexIntoFile.addEntry(fakePHID1, 1, 1, 4, 6);  // Event
    indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 0);  // Lumi
    indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);  // Run
    indexIntoFile.sortVector_Run_Or_Lumi_Entries();

    REQUIRE(!indexIntoFile.empty());

    unsigned count = 0;
    IndexIntoFile::SortedRunOrLumiItr runOrLumi = indexIntoFile.beginRunOrLumi();
    for (IndexIntoFile::SortedRunOrLumiItr endRunOrLumi = indexIntoFile.endRunOrLumi(); runOrLumi != endRunOrLumi;
         ++runOrLumi) {
      long long beginEventNumbers;
      long long endEventNumbers;
      IndexIntoFile::EntryNumber_t beginEventEntry;
      IndexIntoFile::EntryNumber_t endEventEntry;
      runOrLumi.getRange(beginEventNumbers, endEventNumbers, beginEventEntry, endEventEntry);

      if (count == 0) {
        REQUIRE(runOrLumi.isRun());
        REQUIRE(beginEventNumbers == -1);
        REQUIRE(endEventNumbers == -1);
        REQUIRE(beginEventEntry == -1);
        REQUIRE(endEventEntry == -1);
      } else if (count == 3) {
        REQUIRE(!runOrLumi.isRun());
        REQUIRE(beginEventNumbers == 4);
        REQUIRE(endEventNumbers == 7);
        REQUIRE(beginEventEntry == 2);
        REQUIRE(endEventEntry == 5);

        IndexIntoFile::RunOrLumiIndexes const& indexes = runOrLumi.runOrLumiIndexes();
        REQUIRE(indexes.processHistoryIDIndex() == 0);
        REQUIRE(indexes.run() == 1U);
        REQUIRE(indexes.lumi() == 2U);
        REQUIRE(indexes.indexToGetEntry() == 3);
        REQUIRE(indexes.beginEventNumbers() == 4);
        REQUIRE(indexes.endEventNumbers() == 7);
      }

      REQUIRE(runOrLumi.runOrLumi() == count);
      ++count;
    }
    REQUIRE(count == 4U);
    REQUIRE(runOrLumi.runOrLumi() == 4U);
    ++runOrLumi;
    REQUIRE(runOrLumi.runOrLumi() == 4U);

    REQUIRE(runOrLumi == indexIntoFile.endRunOrLumi());
    REQUIRE(!(runOrLumi == indexIntoFile.beginRunOrLumi()));
    REQUIRE(!(iter == indexIntoFile.beginRunOrLumi()));

    REQUIRE(!(runOrLumi != indexIntoFile.endRunOrLumi()));
    REQUIRE(runOrLumi != indexIntoFile.beginRunOrLumi());
    REQUIRE(iter != indexIntoFile.beginRunOrLumi());
  }

  void testKeys() {
    IndexIntoFile::IndexRunKey key1(1, 2);
    REQUIRE(key1.processHistoryIDIndex() == 1);
    REQUIRE(key1.run() == 2);

    IndexIntoFile::IndexRunKey key2(1, 2);
    REQUIRE(!(key1 < key2));
    REQUIRE(!(key2 < key1));

    IndexIntoFile::IndexRunKey key3(1, 3);
    REQUIRE(key1 < key3);
    REQUIRE(!(key3 < key1));

    IndexIntoFile::IndexRunKey key4(10, 1);
    REQUIRE(key1 < key4);
    REQUIRE(!(key4 < key1));

    IndexIntoFile::IndexRunLumiKey k1(1, 2, 3);
    REQUIRE(k1.processHistoryIDIndex() == 1);
    REQUIRE(k1.run() == 2);
    REQUIRE(k1.lumi() == 3);

    IndexIntoFile::IndexRunLumiKey k2(1, 2, 3);
    REQUIRE(!(k1 < k2));
    REQUIRE(!(k2 < k1));

    IndexIntoFile::IndexRunLumiKey k3(1, 2, 4);
    REQUIRE(k1 < k3);
    REQUIRE(!(k3 < k1));

    IndexIntoFile::IndexRunLumiKey k4(1, 3, 1);
    REQUIRE(k1 < k4);
    REQUIRE(!(k4 < k1));

    IndexIntoFile::IndexRunLumiKey k5(11, 1, 1);
    REQUIRE(k1 < k5);
    REQUIRE(!(k5 < k1));

    IndexIntoFile::IndexRunLumiEventKey e1(1, 2, 3, 4);
    REQUIRE(e1.processHistoryIDIndex() == 1);
    REQUIRE(e1.run() == 2);
    REQUIRE(e1.lumi() == 3);
    REQUIRE(e1.event() == 4);

    IndexIntoFile::IndexRunLumiEventKey e2(1, 2, 3, 4);
    REQUIRE(!(e1 < e2));
    REQUIRE(!(e2 < e1));

    IndexIntoFile::IndexRunLumiEventKey e3(1, 2, 3, 5);
    REQUIRE(e1 < e3);
    REQUIRE(!(e3 < e1));

    IndexIntoFile::IndexRunLumiEventKey e4(1, 2, 11, 1);
    REQUIRE(e1 < e4);
    REQUIRE(!(e4 < e1));

    IndexIntoFile::IndexRunLumiEventKey e5(1, 11, 1, 1);
    REQUIRE(e1 < e5);
    REQUIRE(!(e5 < e1));

    IndexIntoFile::IndexRunLumiEventKey e6(11, 1, 1, 1);
    REQUIRE(e1 < e6);
    REQUIRE(!(e6 < e1));
  }

  void testConstructor() {
    edm::IndexIntoFile indexIntoFile;
    REQUIRE(indexIntoFile.runOrLumiEntries().empty());
    REQUIRE(indexIntoFile.processHistoryIDs().empty());
    REQUIRE(indexIntoFile.eventEntries().empty());
    REQUIRE(indexIntoFile.eventNumbers().empty());
    REQUIRE(indexIntoFile.setRunOrLumiEntries().empty());
    REQUIRE(indexIntoFile.setProcessHistoryIDs().empty());

    std::vector<LuminosityBlockNumber_t> lumis;
    lumis.push_back(1);
    edm::IndexIntoFile::IndexIntoFileItr iter = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
    iter.getLumisInRun(lumis);
    REQUIRE(lumis.empty());
  }

};  // TestIndexIntoFile1Fixture

TEST_CASE("IndexIntoFile1", "[indexIntoFile1_t]") {
  TestIndexIntoFile1 fixture;
  fixture.setUp();

  SECTION("testRunOrLumiEntry") { fixture.testRunOrLumiEntry(); }
  SECTION("testRunOrLumiIndexes") { fixture.testRunOrLumiIndexes(); }
  SECTION("testEventEntry") { fixture.testEventEntry(); }
  SECTION("testSortedRunOrLumiItr") { fixture.testSortedRunOrLumiItr(); }
  SECTION("testKeys") { fixture.testKeys(); }
  SECTION("testConstructor") { fixture.testConstructor(); }
}
