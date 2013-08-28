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

class TestIndexIntoFile: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestIndexIntoFile);  
  CPPUNIT_TEST(testIterEndWithLumi);
  CPPUNIT_TEST(testIterEndWithRun);
  CPPUNIT_TEST(testIterLastLumiRangeNoEvents);
  CPPUNIT_TEST(testEmptyIndex);
  CPPUNIT_TEST(testSkip);
  CPPUNIT_TEST(testSkip2);
  CPPUNIT_TEST(testSkip3);
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
    registerProcessHistory(ph1);
    fakePHID1 = ph1.id();

    std::unique_ptr<ProcessHistory> processHistory2(new ProcessHistory);
    ProcessHistory& ph2 = *processHistory2;
    processHistory2->push_back(pc);
    processHistory2->push_back(pc);
    registerProcessHistory(ph2);
    fakePHID2 = ph2.id();

    std::unique_ptr<ProcessHistory> processHistory3(new ProcessHistory);
    ProcessHistory& ph3 = *processHistory3;
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    processHistory3->push_back(pc);
    registerProcessHistory(ph3);
    fakePHID3 = ph3.id();
  }

  void tearDown() { }

  void testIterEndWithLumi();
  void testIterEndWithRun();
  void testIterLastLumiRangeNoEvents();
  void testEmptyIndex();
  void testSkip();
  void testSkip2();
  void testSkip3();
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
  eventEntries.emplace_back(5, 0);
  eventEntries.emplace_back(6, 1);
  eventEntries.emplace_back(7, 2);
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
  eventEntries.emplace_back(1001, 0);
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
  eventEntries.emplace_back(1001, 0);
  eventEntries.emplace_back(1001, 1);
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
  eventEntries.emplace_back(1001, 0);
  indexIntoFile.sortEventEntries();

  edm::IndexIntoFile::IndexIntoFileItr iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);

  skipEventForward(iterNum);
  checkSkipped(0, 2, 102, 0);
  check(iterNum, kRun, 2, 6, -1, 0, 0);
}

void TestIndexIntoFile::testReduce() {
  // This test is implemented in FWCore/Integration/test/ProcessHistory_t.cpp
  // because of dependency issues.
}
