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

class TestIndexIntoFile3: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestIndexIntoFile3);  
  CPPUNIT_TEST(testIterEndWithEvent);
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

  void testIterEndWithEvent();

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

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexIntoFile3);

void TestIndexIntoFile3::check(edm::IndexIntoFile::IndexIntoFileItr const& iter,
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

void TestIndexIntoFile3::checkSkipped(int phIndexOfSkippedEvent,
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

void TestIndexIntoFile3::checkIDRunLumiEntry(edm::IndexIntoFile::IndexIntoFileItr const& iter,
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

void TestIndexIntoFile3::skipEventForward(edm::IndexIntoFile::IndexIntoFileItr & iter) {
  iter.skipEventForward(skipped_.phIndexOfSkippedEvent_,
                        skipped_.runOfSkippedEvent_,
                        skipped_.lumiOfSkippedEvent_,
                        skipped_.skippedEventEntry_);
}

void TestIndexIntoFile3::skipEventBackward(edm::IndexIntoFile::IndexIntoFileItr & iter) {
  iter.skipEventBackward(skipped_.phIndexOfSkippedEvent_,
                         skipped_.runOfSkippedEvent_,
                         skipped_.lumiOfSkippedEvent_,
                         skipped_.skippedEventEntry_);
}

void TestIndexIntoFile3::testIterEndWithEvent() {
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
  eventEntries.emplace_back(7, 0);
  eventEntries.emplace_back(6, 1);
  eventEntries.emplace_back(5, 2);
  eventEntries.emplace_back(4, 3);
  eventEntries.emplace_back(5, 4);
  eventEntries.emplace_back(4, 5);
  eventEntries.emplace_back(4, 6);
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
