/*
 *  indexIntoFile_t.cppunit.cc
 */

#include "cppunit/extensions/HelperMacros.h"

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"

#include "DataFormats/Provenance/interface/IndexIntoFile.h"

#include <string>
#include <iostream>
#include <memory>

using namespace edm;

class TestIndexIntoFile : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestIndexIntoFile);
  CPPUNIT_TEST(testIterEndWithLumi);
  CPPUNIT_TEST(testIterEndWithRun);
  CPPUNIT_TEST(testIterLastLumiRangeNoEvents);
  CPPUNIT_TEST(testEmptyIndex);
  CPPUNIT_TEST(testSkip);
  CPPUNIT_TEST(testSkip2);
  CPPUNIT_TEST(testSkip3);
  CPPUNIT_TEST(testEndWithRun);
  CPPUNIT_TEST(testRunsNoEvents);
  CPPUNIT_TEST(testLumisNoEvents);
  CPPUNIT_TEST_SUITE_END();

public:
  static const IndexIntoFile::EntryType kRun = IndexIntoFile::kRun;
  static const IndexIntoFile::EntryType kLumi = IndexIntoFile::kLumi;
  static const IndexIntoFile::EntryType kEvent = IndexIntoFile::kEvent;
  static const IndexIntoFile::EntryType kEnd = IndexIntoFile::kEnd;

  class Skipped {
  public:
    Skipped() : phIndexOfSkippedEvent_(0), runOfSkippedEvent_(0), lumiOfSkippedEvent_(0), skippedEventEntry_(0) {}
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

  void tearDown() {}

  void testIterEndWithLumi();
  void testIterEndWithRun();
  void testIterLastLumiRangeNoEvents();
  void testEmptyIndex();
  void testSkip();
  void testSkip2();
  void testSkip3();
  void testEndWithRun();
  void testReduce();
  void testRunsNoEvents();
  void testLumisNoEvents();

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

  void skipEventForward(edm::IndexIntoFile::IndexIntoFileItr& iter);
  void skipEventBackward(edm::IndexIntoFile::IndexIntoFileItr& iter);
  void checkSkipped(int phIndexOfSkippedEvent,
                    RunNumber_t runOfSkippedEvent,
                    LuminosityBlockNumber_t lumiOfSkippedEvent,
                    IndexIntoFile::EntryNumber_t skippedEventEntry);
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
  bool theyMatch = iter.getEntryType() == type && iter.type() == type && iter.indexToRun() == indexToRun &&
                   iter.indexToLumi() == indexToLumi && iter.indexToEventRange() == indexToEventRange &&
                   iter.indexToEvent() == indexToEvent && iter.nEvents() == nEvents;
  if (!theyMatch) {
    std::cout << "\nExpected        " << type << "  " << indexToRun << "  " << indexToLumi << "  " << indexToEventRange
              << "  " << indexToEvent << "  " << nEvents << std::endl;
    std::cout << "Iterator values " << iter.type() << "  " << iter.indexToRun() << "  " << iter.indexToLumi() << "  "
              << iter.indexToEventRange() << "  " << iter.indexToEvent() << "  " << iter.nEvents() << std::endl;
  }
  CPPUNIT_ASSERT(theyMatch);
}

void TestIndexIntoFile::checkSkipped(int phIndexOfSkippedEvent,
                                     RunNumber_t runOfSkippedEvent,
                                     LuminosityBlockNumber_t lumiOfSkippedEvent,
                                     IndexIntoFile::EntryNumber_t skippedEventEntry) {
  bool theyMatch =
      skipped_.phIndexOfSkippedEvent_ == phIndexOfSkippedEvent && skipped_.runOfSkippedEvent_ == runOfSkippedEvent &&
      skipped_.lumiOfSkippedEvent_ == lumiOfSkippedEvent && skipped_.skippedEventEntry_ == skippedEventEntry;

  if (!theyMatch) {
    std::cout << "\nExpected        " << phIndexOfSkippedEvent << "  " << runOfSkippedEvent << "  "
              << lumiOfSkippedEvent << "  " << skippedEventEntry << "\n";
    std::cout << "Actual          " << skipped_.phIndexOfSkippedEvent_ << "  " << skipped_.runOfSkippedEvent_ << "  "
              << skipped_.lumiOfSkippedEvent_ << "  " << skipped_.skippedEventEntry_ << "\n";
  }
  CPPUNIT_ASSERT(theyMatch);
}

void TestIndexIntoFile::skipEventForward(edm::IndexIntoFile::IndexIntoFileItr& iter) {
  iter.skipEventForward(skipped_.phIndexOfSkippedEvent_,
                        skipped_.runOfSkippedEvent_,
                        skipped_.lumiOfSkippedEvent_,
                        skipped_.skippedEventEntry_);
}

void TestIndexIntoFile::skipEventBackward(edm::IndexIntoFile::IndexIntoFileItr& iter) {
  iter.skipEventBackward(skipped_.phIndexOfSkippedEvent_,
                         skipped_.runOfSkippedEvent_,
                         skipped_.lumiOfSkippedEvent_,
                         skipped_.skippedEventEntry_);
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

  edm::IndexIntoFile::IndexIntoFileItr iterEntryEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
  CPPUNIT_ASSERT(iterEntryEnd.indexIntoFile() == &indexIntoFile);
  CPPUNIT_ASSERT(iterEntryEnd.size() == 0);
  CPPUNIT_ASSERT(iterEntryEnd.type() == kEnd);
  CPPUNIT_ASSERT(iterEntryEnd.indexToRun() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterEntryEnd.indexToLumi() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterEntryEnd.indexToEventRange() == IndexIntoFile::invalidIndex);
  CPPUNIT_ASSERT(iterEntryEnd.indexToEvent() == 0);
  CPPUNIT_ASSERT(iterEntryEnd.nEvents() == 0);

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

  {
    auto iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
    CPPUNIT_ASSERT(iterEntry == iterEntryEnd);

    skipEventBackward(iterEntry);
    checkSkipped(-1, 0, 0, -1);
    check(iterEntry, kEnd, -1, -1, -1, 0, 0);
  }
}

void TestIndexIntoFile::testIterEndWithLumi() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 0);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 0);    // Run
  indexIntoFile.addEntry(fakePHID1, 12, 101, 0, 1);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 12, 101, 0, 2);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 12, 0, 0, 1);    // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  int i = 0;
  for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
    if (i == 0)
      check(iterFirst, kRun, 0, 1, -1, 0, 0);
    else if (i == 1)
      check(iterFirst, kLumi, 0, 1, -1, 0, 0);
    else if (i == 2)
      check(iterFirst, kRun, 2, 3, -1, 0, 0);
    else if (i == 3)
      check(iterFirst, kLumi, 2, 3, -1, 0, 0);
    else if (i == 4)
      check(iterFirst, kLumi, 2, 4, -1, 0, 0);
    else
      CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 5);

  // Now repeat the above tests for the sorted iteration

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  for (i = 0; iterNum != iterNumEnd; ++iterNum, ++i) {
    if (i == 0)
      check(iterNum, kRun, 0, 1, -1, 0, 0);
    else if (i == 1)
      check(iterNum, kLumi, 0, 1, -1, 0, 0);
    else if (i == 2)
      check(iterNum, kRun, 2, 3, -1, 0, 0);
    else if (i == 3)
      check(iterNum, kLumi, 2, 3, -1, 0, 0);
    else if (i == 4)
      check(iterNum, kLumi, 2, 4, -1, 0, 0);
    else
      CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 5);

  // Now repeat the above tests for the entry iteration

  edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterEntryEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
  for (i = 0; iterEntry != iterEntryEnd; ++iterEntry, ++i) {
    if (i == 0)
      check(iterEntry, kRun, 0, 1, -1, 0, 0);
    else if (i == 1)
      check(iterEntry, kLumi, 0, 1, -1, 0, 0);
    else if (i == 2)
      check(iterEntry, kRun, 2, 3, -1, 0, 0);
    else if (i == 3)
      check(iterEntry, kLumi, 2, 3, -1, 0, 0);
    else if (i == 4)
      check(iterEntry, kLumi, 2, 4, -1, 0, 0);
    else
      CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterEntry.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterEntry.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 5);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);

  skipEventForward(iterEntry);
  checkSkipped(-1, 0, 0, -1);
  check(iterEntry, kEnd, -1, -1, -1, 0, 0);
}

void TestIndexIntoFile::testIterEndWithRun() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);  // Run
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 1);  // Run
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 2);  // Run
  indexIntoFile.addEntry(fakePHID1, 3, 0, 0, 3);  // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  int i = 0;
  for (; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
    if (i == 0)
      check(iterFirst, kRun, 0, -1, -1, 0, 0);
    else if (i == 1)
      check(iterFirst, kRun, 1, -1, -1, 0, 0);
    else if (i == 2)
      check(iterFirst, kRun, 2, -1, -1, 0, 0);
    else if (i == 3)
      check(iterFirst, kRun, 3, -1, -1, 0, 0);
    else
      CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 4);

  // Now repeat the above tests for the sorted iteration

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  i = 0;
  for (; iterNum != iterNumEnd; ++iterNum, ++i) {
    if (i == 0)
      check(iterNum, kRun, 0, -1, -1, 0, 0);
    else if (i == 1)
      check(iterNum, kRun, 1, -1, -1, 0, 0);
    else if (i == 2)
      check(iterNum, kRun, 2, -1, -1, 0, 0);
    else if (i == 3)
      check(iterNum, kRun, 3, -1, -1, 0, 0);
    else
      CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 4);

  // Now repeat the above tests for the entry iteration

  edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterEntryEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
  i = 0;
  for (; iterEntry != iterEntryEnd; ++iterEntry, ++i) {
    if (i == 0)
      check(iterEntry, kRun, 0, -1, -1, 0, 0);
    else if (i == 1)
      check(iterEntry, kRun, 1, -1, -1, 0, 0);
    else if (i == 2)
      check(iterEntry, kRun, 2, -1, -1, 0, 0);
    else if (i == 3)
      check(iterEntry, kRun, 3, -1, -1, 0, 0);
    else
      CPPUNIT_ASSERT(false);

    CPPUNIT_ASSERT(iterEntry.firstEventEntryThisRun() == IndexIntoFile::invalidEntry);
    CPPUNIT_ASSERT(iterEntry.firstEventEntryThisLumi() == IndexIntoFile::invalidEntry);
  }
  CPPUNIT_ASSERT(i == 4);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kRun, 2, -1, -1, 0, 0);
  ++iterFirst;
  check(iterFirst, kRun, 3, -1, -1, 0, 0);
  iterFirst.advanceToNextLumiOrRun();
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kRun, 2, -1, -1, 0, 0);
  ++iterNum;
  check(iterNum, kRun, 3, -1, -1, 0, 0);
  iterNum.advanceToNextLumiOrRun();
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);

  skipEventForward(iterEntry);
  checkSkipped(-1, 0, 0, -1);
  check(iterEntry, kEnd, -1, -1, -1, 0, 0);

  iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
  iterEntry.advanceToNextLumiOrRun();
  check(iterEntry, kRun, 2, -1, -1, 0, 0);
  ++iterEntry;
  check(iterEntry, kRun, 3, -1, -1, 0, 0);
  iterEntry.advanceToNextLumiOrRun();
  check(iterEntry, kEnd, -1, -1, -1, 0, 0);
}

void TestIndexIntoFile::testIterLastLumiRangeNoEvents() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 101, 5, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 101, 0, 0);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 101, 0, 1);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 102, 6, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 102, 0, 2);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 102, 0, 3);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);    // Run
  indexIntoFile.addEntry(fakePHID1, 2, 101, 7, 2);  // Event
  indexIntoFile.addEntry(fakePHID1, 2, 101, 0, 4);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 101, 0, 5);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 1);    // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
  int i = 0;
  for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
    if (i == 0)
      check(iterFirst, kRun, 0, 1, 1, 0, 1);
    else if (i == 1)
      check(iterFirst, kLumi, 0, 1, 1, 0, 1);
    else if (i == 2)
      check(iterFirst, kLumi, 0, 2, 1, 0, 1);
    else if (i == 3)
      check(iterFirst, kEvent, 0, 2, 1, 0, 1);
    else if (i == 4)
      check(iterFirst, kLumi, 0, 3, 3, 0, 1);
    else if (i == 5)
      check(iterFirst, kLumi, 0, 4, 3, 0, 1);
    else if (i == 6)
      check(iterFirst, kEvent, 0, 4, 3, 0, 1);
    else if (i == 7)
      check(iterFirst, kRun, 5, 6, 6, 0, 1);
    else if (i == 8)
      check(iterFirst, kLumi, 5, 6, 6, 0, 1);
    else if (i == 9)
      check(iterFirst, kLumi, 5, 7, 6, 0, 1);
    else if (i == 10)
      check(iterFirst, kEvent, 5, 7, 6, 0, 1);
    else
      CPPUNIT_ASSERT(false);
  }
  CPPUNIT_ASSERT(i == 11);

  std::vector<IndexIntoFile::EventEntry>& eventEntries = indexIntoFile.eventEntries();
  eventEntries.emplace_back(5, 0);
  eventEntries.emplace_back(6, 1);
  eventEntries.emplace_back(7, 2);
  indexIntoFile.sortEventEntries();

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterNumEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
  i = 0;
  for (; iterNum != iterNumEnd; ++iterNum, ++i) {
    if (i == 0)
      check(iterNum, kRun, 0, 1, 1, 0, 1);
    else if (i == 1)
      check(iterNum, kLumi, 0, 1, 1, 0, 1);
    else if (i == 2)
      check(iterNum, kLumi, 0, 2, 1, 0, 1);
    else if (i == 3)
      check(iterNum, kEvent, 0, 2, 1, 0, 1);
    else if (i == 4)
      check(iterNum, kLumi, 0, 3, 3, 0, 1);
    else if (i == 5)
      check(iterNum, kLumi, 0, 4, 3, 0, 1);
    else if (i == 6)
      check(iterNum, kEvent, 0, 4, 3, 0, 1);
    else if (i == 7)
      check(iterNum, kRun, 5, 6, 6, 0, 1);
    else if (i == 8)
      check(iterNum, kLumi, 5, 6, 6, 0, 1);
    else if (i == 9)
      check(iterNum, kLumi, 5, 7, 6, 0, 1);
    else if (i == 10)
      check(iterNum, kEvent, 5, 7, 6, 0, 1);
    else
      CPPUNIT_ASSERT(false);
  }
  CPPUNIT_ASSERT(i == 11);

  edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterEntryEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
  i = 0;
  for (; iterEntry != iterEntryEnd; ++iterEntry, ++i) {
    if (i == 0)
      check(iterEntry, kRun, 0, 1, 1, 0, 1);
    else if (i == 1)
      check(iterEntry, kLumi, 0, 1, 1, 0, 1);
    else if (i == 2)
      check(iterEntry, kLumi, 0, 2, 1, 0, 1);
    else if (i == 3)
      check(iterEntry, kEvent, 0, 2, 1, 0, 1);
    else if (i == 4)
      check(iterEntry, kLumi, 0, 3, 3, 0, 1);
    else if (i == 5)
      check(iterEntry, kLumi, 0, 4, 3, 0, 1);
    else if (i == 6)
      check(iterEntry, kEvent, 0, 4, 3, 0, 1);
    else if (i == 7)
      check(iterEntry, kRun, 5, 6, 6, 0, 1);
    else if (i == 8)
      check(iterEntry, kLumi, 5, 6, 6, 0, 1);
    else if (i == 9)
      check(iterEntry, kLumi, 5, 7, 6, 0, 1);
    else if (i == 10)
      check(iterEntry, kEvent, 5, 7, 6, 0, 1);
    else
      CPPUNIT_ASSERT(false);
  }
  CPPUNIT_ASSERT(i == 11);

  skipEventBackward(iterEntry);
  checkSkipped(0, 2, 101, 2);
}

void TestIndexIntoFile::testSkip() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 101, 1001, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 101, 0, 0);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 101, 0, 1);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 102, 0, 2);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);       // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kRun, 0, 3, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  skipEventBackward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kRun, 0, 1, 1, 0, 1);

  skipEventBackward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kRun, 0, 1, 1, 0, 1);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  skipEventForward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kLumi, 0, 3, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  std::vector<IndexIntoFile::EventEntry>& eventEntries = indexIntoFile.eventEntries();
  eventEntries.emplace_back(1001, 0);
  indexIntoFile.sortEventEntries();

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kRun, 0, 3, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  skipEventBackward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kRun, 0, 1, 1, 0, 1);

  skipEventBackward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kRun, 0, 1, 1, 0, 1);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  skipEventForward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kLumi, 0, 3, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0, 0);
  {
    edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);

    skipEventForward(iterEntry);
    checkSkipped(0, 1, 101, 0);
    check(iterEntry, kRun, 0, 3, -1, 0, 0);

    skipEventForward(iterEntry);
    checkSkipped(-1, 0, 0, -1);
    check(iterEntry, kEnd, -1, -1, -1, 0, 0);

    skipEventBackward(iterEntry);
    checkSkipped(0, 1, 101, 0);
    check(iterEntry, kRun, 0, 1, 1, 0, 1);

    skipEventBackward(iterEntry);
    checkSkipped(-1, 0, 0, -1);
    check(iterEntry, kRun, 0, 1, 1, 0, 1);

    iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
    ++iterEntry;
    skipEventForward(iterEntry);
    checkSkipped(0, 1, 101, 0);
    check(iterEntry, kLumi, 0, 3, -1, 0, 0);

    skipEventForward(iterEntry);
    checkSkipped(-1, 0, 0, -1);
    check(iterEntry, kEnd, -1, -1, -1, 0, 0);
  }
}

void TestIndexIntoFile::testSkip2() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 101, 1001, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 101, 0, 0);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 101, 0, 1);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 102, 0, 2);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);       // Run
  indexIntoFile.addEntry(fakePHID1, 2, 101, 1001, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 2, 101, 0, 3);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 101, 0, 4);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 102, 0, 5);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 1);       // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kRun, 0, 3, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(0, 2, 101, 1);
  check(iterFirst, kRun, 4, 7, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  skipEventBackward(iterFirst);
  checkSkipped(0, 2, 101, 1);
  check(iterFirst, kRun, 4, 5, 5, 0, 1);

  skipEventBackward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kRun, 0, 1, 1, 0, 1);

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  skipEventForward(iterFirst);
  checkSkipped(0, 1, 101, 0);
  check(iterFirst, kLumi, 0, 3, -1, 0, 0);

  skipEventForward(iterFirst);
  checkSkipped(0, 2, 101, 1);
  check(iterFirst, kRun, 4, 7, -1, 0, 0);

  std::vector<IndexIntoFile::EventEntry>& eventEntries = indexIntoFile.eventEntries();
  eventEntries.emplace_back(1001, 0);
  eventEntries.emplace_back(1001, 1);
  indexIntoFile.sortEventEntries();

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kRun, 0, 3, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(0, 2, 101, 1);
  check(iterNum, kRun, 4, 7, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  skipEventBackward(iterNum);
  checkSkipped(0, 2, 101, 1);
  check(iterNum, kRun, 4, 5, 5, 0, 1);

  skipEventBackward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kRun, 0, 1, 1, 0, 1);

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  skipEventForward(iterNum);
  checkSkipped(0, 1, 101, 0);
  check(iterNum, kLumi, 0, 3, -1, 0, 0);

  skipEventForward(iterNum);
  checkSkipped(0, 2, 101, 1);
  check(iterNum, kRun, 4, 7, -1, 0, 0);

  {
    edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);

    skipEventForward(iterEntry);
    checkSkipped(0, 1, 101, 0);
    check(iterEntry, kRun, 0, 3, -1, 0, 0);

    skipEventForward(iterEntry);
    checkSkipped(0, 2, 101, 1);
    check(iterEntry, kRun, 4, 7, -1, 0, 0);

    skipEventForward(iterEntry);
    checkSkipped(-1, 0, 0, -1);
    check(iterEntry, kEnd, -1, -1, -1, 0, 0);

    skipEventBackward(iterEntry);
    checkSkipped(0, 2, 101, 1);
    check(iterEntry, kRun, 4, 5, 5, 0, 1);

    skipEventBackward(iterEntry);
    checkSkipped(0, 1, 101, 0);
    check(iterEntry, kRun, 0, 1, 1, 0, 1);

    iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
    ++iterEntry;
    skipEventForward(iterEntry);
    checkSkipped(0, 1, 101, 0);
    check(iterEntry, kLumi, 0, 3, -1, 0, 0);

    skipEventForward(iterEntry);
    checkSkipped(0, 2, 101, 1);
    check(iterEntry, kRun, 4, 7, -1, 0, 0);
  }
}

void TestIndexIntoFile::testSkip3() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 0);       // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);       // Run
  indexIntoFile.addEntry(fakePHID1, 2, 101, 0, 1);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 101, 0, 2);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 102, 1001, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 2, 102, 0, 3);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 103, 0, 4);     // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 1);       // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);

  skipEventForward(iterFirst);
  checkSkipped(0, 2, 102, 0);
  check(iterFirst, kRun, 2, 6, -1, 0, 0);

  std::vector<IndexIntoFile::EventEntry>& eventEntries = indexIntoFile.eventEntries();
  eventEntries.emplace_back(1001, 0);
  indexIntoFile.sortEventEntries();

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(0, 2, 102, 0);
  check(iterNum, kRun, 2, 6, -1, 0, 0);

  {
    edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);

    skipEventForward(iterEntry);
    checkSkipped(0, 2, 102, 0);
    check(iterEntry, kRun, 2, 6, -1, 0, 0);
  }
}

void TestIndexIntoFile::testEndWithRun() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);  // Run

  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  check(iterFirst, kRun, 0, -1, -1, 0, 0);
  ++iterFirst;
  check(iterFirst, kEnd, -1, -1, -1, 0, 0);

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  check(iterNum, kRun, 0, -1, -1, 0, 0);
  ++iterNum;
  check(iterNum, kEnd, -1, -1, -1, 0, 0);

  {
    edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
    check(iterEntry, kRun, 0, -1, -1, 0, 0);
    ++iterEntry;
    check(iterEntry, kEnd, -1, -1, -1, 0, 0);
  }
}

void TestIndexIntoFile::testReduce() {
  // This test is implemented in FWCore/Integration/test/ProcessHistory_t.cpp
  // because of dependency issues.
}

void TestIndexIntoFile::testRunsNoEvents() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 8, 1, 0, 0);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 5, 1, 0, 1);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 2);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 3, 1, 1, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 3, 1, 0, 3);  // Lumi

  indexIntoFile.addEntry(fakePHID1, 4, 1, 1, 1);   // Event
  indexIntoFile.addEntry(fakePHID1, 4, 1, 0, 4);   // Lumi
  indexIntoFile.addEntry(fakePHID1, 7, 1, 0, 5);   // Lumi
  indexIntoFile.addEntry(fakePHID1, 7, 2, 0, 6);   // Lumi
  indexIntoFile.addEntry(fakePHID1, 7, 3, 1, 2);   // Event
  indexIntoFile.addEntry(fakePHID1, 7, 3, 0, 7);   // Lumi
  indexIntoFile.addEntry(fakePHID1, 7, 4, 0, 8);   // Lumi
  indexIntoFile.addEntry(fakePHID1, 7, 5, 0, 9);   // Lumi
  indexIntoFile.addEntry(fakePHID1, 7, 6, 1, 3);   // Event
  indexIntoFile.addEntry(fakePHID1, 7, 6, 0, 10);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 7, 7, 1, 4);   // Event
  indexIntoFile.addEntry(fakePHID1, 7, 7, 0, 11);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 7, 8, 0, 12);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 7, 9, 0, 13);  // Lumi

  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);  // Run
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 1);  // Run
  indexIntoFile.addEntry(fakePHID1, 3, 0, 0, 2);  // Run
  indexIntoFile.addEntry(fakePHID1, 4, 0, 0, 3);  // Run
  indexIntoFile.addEntry(fakePHID1, 5, 0, 0, 4);  // Run
  indexIntoFile.addEntry(fakePHID1, 6, 0, 0, 5);  // Run
  indexIntoFile.addEntry(fakePHID1, 7, 0, 0, 6);  // Run
  indexIntoFile.addEntry(fakePHID1, 8, 0, 0, 7);  // Run
  indexIntoFile.addEntry(fakePHID1, 9, 0, 0, 8);  // Run

  indexIntoFile.addEntry(fakePHID1, 8, 0, 0, 9);   // Run
  indexIntoFile.addEntry(fakePHID1, 5, 0, 0, 10);  // Run
  indexIntoFile.addEntry(fakePHID1, 4, 0, 0, 11);  // Run
  indexIntoFile.addEntry(fakePHID1, 4, 0, 0, 12);  // Run
  indexIntoFile.addEntry(fakePHID1, 4, 0, 0, 13);  // Run
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 14);  // Run
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 15);  // Run

  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterEntryEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
  int i = 0;
  for (; iterEntry != iterEntryEnd; ++iterEntry, ++i) {
    if (i == 0) {
      check(iterEntry, kRun, 0, 1, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 1 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
      CPPUNIT_ASSERT(iterEntry.shouldProcessRun());
    } else if (i == 1) {
      check(iterEntry, kLumi, 0, 1, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 1 && iterEntry.lumi() == 1);
    } else if (i == 2) {
      check(iterEntry, kRun, 2, -1, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 3) {
      check(iterEntry, kRun, 3, -1, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 4) {
      check(iterEntry, kRun, 4, -1, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 5) {
      check(iterEntry, kRun, 5, 6, 6, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 3 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 6) {
      check(iterEntry, kLumi, 5, 6, 6, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 3 && iterEntry.lumi() == 1);
    } else if (i == 7) {
      check(iterEntry, kEvent, 5, 6, 6, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 3 && iterEntry.lumi() == 1);
    } else if (i == 8) {
      check(iterEntry, kRun, 7, 11, 11, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 4 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 9) {
      check(iterEntry, kRun, 8, 11, 11, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 4 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 10) {
      check(iterEntry, kRun, 9, 11, 11, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 4 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 11) {
      check(iterEntry, kRun, 10, 11, 11, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 4 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 12) {
      check(iterEntry, kLumi, 10, 11, 11, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 4 && iterEntry.lumi() == 1);
    } else if (i == 13) {
      check(iterEntry, kEvent, 10, 11, 11, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 4 && iterEntry.lumi() == 1);
    } else if (i == 14) {
      check(iterEntry, kRun, 12, 14, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 5 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 15) {
      check(iterEntry, kRun, 13, 14, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 5 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 16) {
      check(iterEntry, kLumi, 13, 14, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 5 && iterEntry.lumi() == 1);
    } else if (i == 17) {
      check(iterEntry, kRun, 15, -1, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 6 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 18) {
      check(iterEntry, kRun, 16, 17, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 19) {
      check(iterEntry, kLumi, 16, 17, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 1);
    } else if (i == 20) {
      check(iterEntry, kLumi, 16, 18, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 2);
    } else if (i == 21) {
      check(iterEntry, kLumi, 16, 19, 19, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 3);
    } else if (i == 22) {
      check(iterEntry, kEvent, 16, 19, 19, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 3);
    } else if (i == 23) {
      check(iterEntry, kLumi, 16, 20, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 4);
    } else if (i == 24) {
      check(iterEntry, kLumi, 16, 21, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 5);
    } else if (i == 25) {
      check(iterEntry, kLumi, 16, 22, 22, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 6);
    } else if (i == 26) {
      check(iterEntry, kEvent, 16, 22, 22, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 6);
    } else if (i == 27) {
      check(iterEntry, kLumi, 16, 23, 23, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 7);
    } else if (i == 28) {
      check(iterEntry, kEvent, 16, 23, 23, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 7);
    } else if (i == 29) {
      check(iterEntry, kLumi, 16, 24, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 8);
    } else if (i == 30) {
      check(iterEntry, kLumi, 16, 25, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 7 && iterEntry.lumi() == 9);
    } else if (i == 31) {
      check(iterEntry, kRun, 26, 28, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 8 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 32) {
      check(iterEntry, kRun, 27, 28, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 8 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else if (i == 33) {
      check(iterEntry, kLumi, 27, 28, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 8 && iterEntry.lumi() == 1);
    } else if (i == 34) {
      check(iterEntry, kRun, 29, -1, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 9 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
    } else
      CPPUNIT_ASSERT(false);
  }
  CPPUNIT_ASSERT(i == 35);
}

void TestIndexIntoFile::testLumisNoEvents() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 2, 7, 1, 0);  // Event 2:7:1
  indexIntoFile.addEntry(fakePHID1, 2, 8, 1, 1);  // Event 2:8:1

  indexIntoFile.addEntry(fakePHID1, 1, 1, 1, 2);  // Event 1:1:1
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 0);  // Lumi  1:1
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);  // Run   1

  indexIntoFile.addEntry(fakePHID1, 2, 7, 2, 3);  // Event 2:7:2

  indexIntoFile.addEntry(fakePHID1, 2, 1, 1, 4);  // Event 2:1:1
  indexIntoFile.addEntry(fakePHID1, 2, 1, 0, 1);  // Lumi  2:1
  indexIntoFile.addEntry(fakePHID1, 2, 2, 0, 2);  // Lumi  2:2
  indexIntoFile.addEntry(fakePHID1, 2, 3, 0, 3);  // Lumi  2:3
  indexIntoFile.addEntry(fakePHID1, 2, 4, 0, 4);  // Lumi  2:4
  indexIntoFile.addEntry(fakePHID1, 2, 5, 1, 5);  // Event 2:5:1
  indexIntoFile.addEntry(fakePHID1, 2, 5, 0, 5);  // Lumi  2:5
  indexIntoFile.addEntry(fakePHID1, 2, 6, 0, 6);  // Lumi  2:6
  indexIntoFile.addEntry(fakePHID1, 2, 7, 3, 6);  // Event 2:7:3
  indexIntoFile.addEntry(fakePHID1, 2, 7, 0, 7);  // Lumi  2:7
  indexIntoFile.addEntry(fakePHID1, 2, 8, 1, 7);  // Event 2:8:1
  indexIntoFile.addEntry(fakePHID1, 2, 8, 0, 8);  // Lumi  2:8
  indexIntoFile.addEntry(fakePHID1, 2, 9, 0, 9);  // Lumi  2:9
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 1);  // Run   2

  indexIntoFile.addEntry(fakePHID1, 3, 1, 1, 8);   // Event 3:1:1
  indexIntoFile.addEntry(fakePHID1, 3, 1, 0, 10);  // Lumi  3:1
  indexIntoFile.addEntry(fakePHID1, 3, 0, 0, 2);   // Run   3

  indexIntoFile.addEntry(fakePHID1, 2, 4, 1, 9);    // Event 2:4:1
  indexIntoFile.addEntry(fakePHID1, 2, 7, 4, 10);   // Event 2:7:4
  indexIntoFile.addEntry(fakePHID1, 2, 7, 5, 11);   // Event 2:7:5
  indexIntoFile.addEntry(fakePHID1, 2, 7, 0, 11);   // Lumi  2:7
  indexIntoFile.addEntry(fakePHID1, 2, 3, 1, 12);   // Event 2:3:1
  indexIntoFile.addEntry(fakePHID1, 2, 3, 0, 12);   // Lumi  2:3
  indexIntoFile.addEntry(fakePHID1, 2, 4, 2, 13);   // Event 2:4:1
  indexIntoFile.addEntry(fakePHID1, 2, 4, 0, 13);   // Lumi  2:4
  indexIntoFile.addEntry(fakePHID1, 2, 7, 6, 14);   // Event 2:7:6
  indexIntoFile.addEntry(fakePHID1, 2, 7, 0, 14);   // Lumi  2:7
  indexIntoFile.addEntry(fakePHID1, 2, 7, 7, 15);   // Event 2:7:7
  indexIntoFile.addEntry(fakePHID1, 2, 7, 8, 16);   // Event 2:7:8
  indexIntoFile.addEntry(fakePHID1, 2, 3, 0, 15);   // Lumi  2:3
  indexIntoFile.addEntry(fakePHID1, 2, 7, 9, 17);   // Event 2:7:9
  indexIntoFile.addEntry(fakePHID1, 2, 7, 10, 18);  // Event 2:7:10
  indexIntoFile.addEntry(fakePHID1, 2, 7, 0, 16);   // Lumi  2:7
  indexIntoFile.addEntry(fakePHID1, 2, 7, 0, 17);   // Lumi  2:7
  indexIntoFile.addEntry(fakePHID1, 2, 7, 0, 18);   // Lumi  2:7
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 3);    // Run   2
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 4);    // Run   2
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterEntryEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
  int i = 0;
  for (; iterEntry != iterEntryEnd; ++iterEntry, ++i) {
    if (i == 0) {
      check(iterEntry, kRun, 0, 1, 1, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessRun());
      CPPUNIT_ASSERT(iterEntry.entry() == 1);
    } else if (i == 1) {
      check(iterEntry, kLumi, 0, 1, 1, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 7);
    } else if (i == 2) {
      check(iterEntry, kEvent, 0, 1, 1, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 0);
    } else if (i == 3) {
      check(iterEntry, kLumi, 0, 2, 2, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 8);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 8);
    } else if (i == 4) {
      check(iterEntry, kEvent, 0, 2, 2, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 8);
      CPPUNIT_ASSERT(iterEntry.entry() == 1);
    } else if (i == 5) {
      check(iterEntry, kRun, 3, 4, 4, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 1 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
      CPPUNIT_ASSERT(iterEntry.shouldProcessRun());
      CPPUNIT_ASSERT(iterEntry.entry() == 0);
    } else if (i == 6) {
      check(iterEntry, kLumi, 3, 4, 4, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 1 && iterEntry.lumi() == 1);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 0);
    } else if (i == 7) {
      check(iterEntry, kEvent, 3, 4, 4, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 1 && iterEntry.lumi() == 1);
      CPPUNIT_ASSERT(iterEntry.entry() == 2);
    } else if (i == 8) {
      check(iterEntry, kRun, 5, 6, 6, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessRun());
      CPPUNIT_ASSERT(iterEntry.entry() == 1);
    } else if (i == 9) {
      check(iterEntry, kLumi, 5, 6, 6, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 7);
    } else if (i == 10) {
      check(iterEntry, kEvent, 5, 6, 6, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 3);
    } else if (i == 11) {
      check(iterEntry, kLumi, 5, 7, 7, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 1);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 1);
    } else if (i == 12) {
      check(iterEntry, kEvent, 5, 7, 7, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 1);
      CPPUNIT_ASSERT(iterEntry.entry() == 4);
    } else if (i == 13) {
      check(iterEntry, kLumi, 5, 8, 8, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 5);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 5);
    } else if (i == 14) {
      check(iterEntry, kEvent, 5, 8, 8, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 5);
      CPPUNIT_ASSERT(iterEntry.entry() == 5);
    } else if (i == 15) {
      check(iterEntry, kLumi, 5, 9, 9, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 7);
    } else if (i == 16) {
      check(iterEntry, kEvent, 5, 9, 9, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 6);
    } else if (i == 17) {
      check(iterEntry, kLumi, 5, 10, 10, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 8);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 8);
    } else if (i == 18) {
      check(iterEntry, kEvent, 5, 10, 10, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 8);
      CPPUNIT_ASSERT(iterEntry.entry() == 7);
    } else if (i == 19) {
      check(iterEntry, kRun, 11, 12, 12, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 3 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
      CPPUNIT_ASSERT(iterEntry.shouldProcessRun());
      CPPUNIT_ASSERT(iterEntry.entry() == 2);
    } else if (i == 20) {
      check(iterEntry, kLumi, 11, 12, 12, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 3 && iterEntry.lumi() == 1);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 10);
    } else if (i == 21) {
      check(iterEntry, kEvent, 11, 12, 12, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 3 && iterEntry.lumi() == 1);
      CPPUNIT_ASSERT(iterEntry.entry() == 8);
    } else if (i == 22) {
      check(iterEntry, kRun, 13, 16, 16, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
      CPPUNIT_ASSERT(iterEntry.shouldProcessRun());
      CPPUNIT_ASSERT(iterEntry.entry() == 1);
    } else if (i == 23) {
      check(iterEntry, kRun, 14, 16, 16, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
      CPPUNIT_ASSERT(iterEntry.shouldProcessRun());
      CPPUNIT_ASSERT(iterEntry.entry() == 3);
    } else if (i == 24) {
      check(iterEntry, kRun, 15, 16, 16, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == IndexIntoFile::invalidLumi);
      CPPUNIT_ASSERT(iterEntry.shouldProcessRun());
      CPPUNIT_ASSERT(iterEntry.entry() == 4);
    } else if (i == 25) {
      check(iterEntry, kLumi, 15, 16, 16, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 4);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 4);
    } else if (i == 26) {
      check(iterEntry, kEvent, 15, 16, 16, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 4);
      CPPUNIT_ASSERT(iterEntry.entry() == 9);
    } else if (i == 27) {
      check(iterEntry, kLumi, 15, 17, 17, 0, 2);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 11);
    } else if (i == 28) {
      check(iterEntry, kEvent, 15, 17, 17, 0, 2);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 10);
    } else if (i == 29) {
      check(iterEntry, kEvent, 15, 17, 17, 1, 2);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 11);
    } else if (i == 30) {
      check(iterEntry, kLumi, 15, 18, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 1);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 1);
    } else if (i == 31) {
      check(iterEntry, kLumi, 15, 19, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 2);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 2);
    } else if (i == 32) {
      check(iterEntry, kLumi, 15, 20, 21, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 3);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 3);
    } else if (i == 33) {
      check(iterEntry, kLumi, 15, 21, 21, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 3);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 12);
    } else if (i == 34) {
      check(iterEntry, kLumi, 15, 22, 21, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 3);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 15);
    } else if (i == 35) {
      check(iterEntry, kEvent, 15, 22, 21, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 3);
      CPPUNIT_ASSERT(iterEntry.entry() == 12);
    } else if (i == 36) {
      check(iterEntry, kLumi, 15, 23, 24, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 4);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 4);
    } else if (i == 37) {
      check(iterEntry, kLumi, 15, 24, 24, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 4);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 13);
    } else if (i == 38) {
      check(iterEntry, kEvent, 15, 24, 24, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 4);
      CPPUNIT_ASSERT(iterEntry.entry() == 13);
    } else if (i == 39) {
      check(iterEntry, kLumi, 15, 25, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 5);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 5);
    } else if (i == 40) {
      check(iterEntry, kLumi, 15, 26, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 6);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 6);
    } else if (i == 41) {
      check(iterEntry, kLumi, 15, 27, 29, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 7);
    } else if (i == 42) {
      check(iterEntry, kLumi, 15, 28, 29, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 11);
    } else if (i == 43) {
      check(iterEntry, kLumi, 15, 29, 29, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 14);
    } else if (i == 44) {
      check(iterEntry, kLumi, 15, 30, 29, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(!iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 16);
    } else if (i == 45) {
      check(iterEntry, kLumi, 15, 31, 29, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 16);
    } else if (i == 46) {
      check(iterEntry, kLumi, 15, 32, 29, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 17);
    } else if (i == 47) {
      check(iterEntry, kLumi, 15, 33, 29, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 18);
    } else if (i == 48) {
      check(iterEntry, kEvent, 15, 33, 29, 0, 1);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 14);
    } else if (i == 49) {
      check(iterEntry, kEvent, 15, 33, 30, 0, 2);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 15);
    } else if (i == 50) {
      check(iterEntry, kEvent, 15, 33, 30, 1, 2);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 16);
    } else if (i == 51) {
      check(iterEntry, kEvent, 15, 33, 31, 0, 2);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 17);
    } else if (i == 52) {
      check(iterEntry, kEvent, 15, 33, 31, 1, 2);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 7);
      CPPUNIT_ASSERT(iterEntry.entry() == 18);
    } else if (i == 53) {
      check(iterEntry, kLumi, 15, 34, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 8);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 8);
    } else if (i == 54) {
      check(iterEntry, kLumi, 15, 35, -1, 0, 0);
      CPPUNIT_ASSERT(iterEntry.run() == 2 && iterEntry.lumi() == 9);
      CPPUNIT_ASSERT(iterEntry.shouldProcessLumi());
      CPPUNIT_ASSERT(iterEntry.entry() == 9);
    } else
      CPPUNIT_ASSERT(false);
  }
  CPPUNIT_ASSERT(iterEntry.entry() == IndexIntoFile::invalidEntry);
  CPPUNIT_ASSERT(i == 55);

  skipEventBackward(iterEntry);
  checkSkipped(0, 2, 7, 18);
  check(iterEntry, kRun, 13, 27, 31, 1, 2);
}
