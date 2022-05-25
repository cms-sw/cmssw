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

class TestIndexIntoFile3 : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestIndexIntoFile3);
  CPPUNIT_TEST(testIterEndWithEvent);
  CPPUNIT_TEST(testOverlappingLumis);
  CPPUNIT_TEST(testOverlappingLumisMore);
  CPPUNIT_TEST(testOverlappingLumisOutOfOrderEvent);
  CPPUNIT_TEST(testOverlappingLumisWithEndWithEmptyLumi);
  CPPUNIT_TEST(testOverlappingLumisWithLumiEndOrderChanged);
  CPPUNIT_TEST(testNonContiguousRun);
  CPPUNIT_TEST(testNonValidLumiInsideValidLumis);
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

  void testIterEndWithEvent();
  void testOverlappingLumis();
  void testOverlappingLumisMore();
  void testOverlappingLumisOutOfOrderEvent();
  void testOverlappingLumisWithEndWithEmptyLumi();
  void testOverlappingLumisWithLumiEndOrderChanged();
  void testNonContiguousRun();
  void testNonValidLumiInsideValidLumis();

  ProcessHistoryID nullPHID;
  ProcessHistoryID fakePHID1;
  ProcessHistoryID fakePHID2;
  ProcessHistoryID fakePHID3;

  bool check(edm::IndexIntoFile::IndexIntoFileItr const& iter,
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

  void checkIDRunLumiEntry(edm::IndexIntoFile::IndexIntoFileItr const& iter,
                           int phIndex,
                           RunNumber_t run,
                           LuminosityBlockNumber_t lumi,
                           IndexIntoFile::EntryNumber_t entry);
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexIntoFile3);

bool TestIndexIntoFile3::check(edm::IndexIntoFile::IndexIntoFileItr const& iter,
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
              << "  " << indexToEvent << "  " << nEvents << "\n";
    std::cout << "Iterator values " << iter.type() << "  " << iter.indexToRun() << "  " << iter.indexToLumi() << "  "
              << iter.indexToEventRange() << "  " << iter.indexToEvent() << "  " << iter.nEvents() << "\n";
  }
  return theyMatch;
}

void TestIndexIntoFile3::checkSkipped(int phIndexOfSkippedEvent,
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

void TestIndexIntoFile3::checkIDRunLumiEntry(edm::IndexIntoFile::IndexIntoFileItr const& iter,
                                             int phIndex,
                                             RunNumber_t run,
                                             LuminosityBlockNumber_t lumi,
                                             IndexIntoFile::EntryNumber_t entry) {
  bool theyMatch =
      iter.processHistoryIDIndex() == phIndex && iter.run() == run && iter.lumi() == lumi && iter.entry() == entry;

  if (!theyMatch) {
    std::cout << "\nExpected        " << phIndex << "  " << run << "  " << lumi << "  " << entry << "\n";
    std::cout << "Actual          " << iter.processHistoryIDIndex() << "  " << iter.run() << "  " << iter.lumi() << "  "
              << iter.entry() << "\n";
  }
  CPPUNIT_ASSERT(theyMatch);
}

void TestIndexIntoFile3::skipEventForward(edm::IndexIntoFile::IndexIntoFileItr& iter) {
  iter.skipEventForward(skipped_.phIndexOfSkippedEvent_,
                        skipped_.runOfSkippedEvent_,
                        skipped_.lumiOfSkippedEvent_,
                        skipped_.skippedEventEntry_);
}

void TestIndexIntoFile3::skipEventBackward(edm::IndexIntoFile::IndexIntoFileItr& iter) {
  iter.skipEventBackward(skipped_.phIndexOfSkippedEvent_,
                         skipped_.runOfSkippedEvent_,
                         skipped_.lumiOfSkippedEvent_,
                         skipped_.skippedEventEntry_);
}

void TestIndexIntoFile3::testIterEndWithEvent() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 11, 101, 7, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 6, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 0);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 1);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 101, 5, 2);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 4, 3);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 2);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 102, 5, 4);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 102, 4, 5);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 102, 0, 3);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 0);    // Run
  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 1);    // Run
  indexIntoFile.addEntry(fakePHID2, 11, 101, 0, 4);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 5);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 4, 6);  // Event
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 6);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 2);    // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  edm::IndexIntoFile::IndexIntoFileItr iter3(
      &indexIntoFile, IndexIntoFile::firstAppearanceOrder, IndexIntoFile::kEvent, 0, 3, 2, 1, 2);
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
    CPPUNIT_ASSERT(iterFirst == iterFirstCopy);
    iterFirstCopy2 = iterFirstCopy;
    CPPUNIT_ASSERT(iterFirst == iterFirstCopy2);
    if (i == 0) {
      CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 1, 0, 2));
      CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterFirst.size() == 10);
      CPPUNIT_ASSERT(iterFirst.indexedSize() == 10);
      CPPUNIT_ASSERT(iterFirst.shouldProcessRun());
    } else if (i == 1) {
      CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 1, 0, 2));
      CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
    } else if (i == 2)
      CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 2, 1, 0, 2));
    else if (i == 3)
      CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 3, 1, 0, 2));
    else if (i == 4)
      CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 1, 0, 2));
    else if (i == 5)
      CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 1, 1, 2));
    else if (i == 6)
      CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 3, 0, 2));
    else if (i == 7)
      CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 3, 1, 2));
    else if (i == 8)
      CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 0, 2));
    else if (i == 9)
      CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 0, 2));
    else if (i == 10)
      CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 1, 2));
    else if (i == 11)
      CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));
    else if (i == 12)
      CPPUNIT_ASSERT(check(iterFirst, kRun, 6, 7, -1, 0, 0));
    else if (i == 13)
      CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 7, -1, 0, 0));
    else if (i == 14)
      CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 8, 9, 0, 1));
    else if (i == 15)
      CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 9, 9, 0, 1));
    else if (i == 16)
      CPPUNIT_ASSERT(check(iterFirst, kEvent, 6, 9, 9, 0, 1));
    else
      CPPUNIT_ASSERT(false);

    switch (i) {
      case 0:
        CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
        break;
      case 10:
        CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
        break;
      case 12:
        CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 6);
        break;
    }

    switch (i) {
      case 0:
        CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
        break;
      case 1:
        CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
        break;
      case 2:
        CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
        break;
      case 3:
        CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
        break;
      case 10:
        CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 4);
        break;
      case 12:
        CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
        break;
    }
  }
  CPPUNIT_ASSERT(i == 17);

  for (i = 0, iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder); iterFirst != iterFirstEnd;
       ++iterFirst, ++i) {
    switch (i) {
      case 0:
        checkIDRunLumiEntry(iterFirst, 0, 11, 0, 0);  // Run
        break;
      case 1:
        checkIDRunLumiEntry(iterFirst, 0, 11, 101, 0);  // Lumi
        break;
      case 2:
        checkIDRunLumiEntry(iterFirst, 0, 11, 101, 1);  // Lumi
        break;
      case 3:
        checkIDRunLumiEntry(iterFirst, 0, 11, 101, 2);  // Lumi
        break;
      case 4:
        checkIDRunLumiEntry(iterFirst, 0, 11, 101, 0);  // Event
        break;
      case 5:
        checkIDRunLumiEntry(iterFirst, 0, 11, 101, 1);  // Event
        break;
      case 6:
        checkIDRunLumiEntry(iterFirst, 0, 11, 101, 2);  // Event
        break;
      case 7:
        checkIDRunLumiEntry(iterFirst, 0, 11, 101, 3);  // Event
        break;
      case 8:
        checkIDRunLumiEntry(iterFirst, 0, 11, 102, 3);  // Lumi
        break;
      case 9:
        checkIDRunLumiEntry(iterFirst, 0, 11, 102, 4);  // Event
        break;
      case 10:
        checkIDRunLumiEntry(iterFirst, 0, 11, 102, 5);  // Event
        break;
      case 11:
        checkIDRunLumiEntry(iterFirst, 1, 11, 0, 1);  // Run
        break;
      case 12:
        checkIDRunLumiEntry(iterFirst, 1, 11, 0, 2);  // Run
        break;
      case 13:
        checkIDRunLumiEntry(iterFirst, 1, 11, 101, 4);  // Lumi
        break;
      case 14:
        checkIDRunLumiEntry(iterFirst, 1, 11, 102, 5);  // Lumi
        break;
      case 15:
        checkIDRunLumiEntry(iterFirst, 1, 11, 102, 6);  // Lumi
        break;
      case 16:
        checkIDRunLumiEntry(iterFirst, 1, 11, 102, 6);  // Event
        break;
    }
  }
  checkIDRunLumiEntry(iterFirst, -1, 0, 0, -1);  // Event

  CPPUNIT_ASSERT(indexIntoFile.runOrLumiIndexes().empty());

  // Now repeat the above tests for the sorted iteration

  edm::IndexIntoFile::IndexIntoFileItr iter4(
      &indexIntoFile, IndexIntoFile::numericalOrder, IndexIntoFile::kEvent, 0, 3, 1, 3, 4);
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

  std::vector<IndexIntoFile::EventEntry>& eventEntries = indexIntoFile.eventEntries();
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
      CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 0, 4));
      CPPUNIT_ASSERT(iterNum.indexIntoFile() == &indexIntoFile);
      CPPUNIT_ASSERT(iterNum.size() == 10);
      CPPUNIT_ASSERT(iterNum.indexedSize() == 10);
      CPPUNIT_ASSERT(iterNum.shouldProcessRun());
    } else if (i == 1) {
      CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 1, 1, 0, 4));
      CPPUNIT_ASSERT(iterNum.shouldProcessLumi());
    } else if (i == 2)
      CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 2, 1, 0, 4));
    else if (i == 3)
      CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 3, 1, 0, 4));
    else if (i == 4)
      CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 3, 1, 0, 4));
    else if (i == 5)
      CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 3, 1, 1, 4));
    else if (i == 6)
      CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 3, 1, 2, 4));
    else if (i == 7)
      CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 3, 1, 3, 4));
    else if (i == 8)
      CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 4, 4, 0, 2));
    else if (i == 9)
      CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 4, 4, 0, 2));
    else if (i == 10)
      CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 4, 4, 1, 2));
    else if (i == 11)
      CPPUNIT_ASSERT(check(iterNum, kRun, 5, 7, -1, 0, 0));
    else if (i == 12)
      CPPUNIT_ASSERT(check(iterNum, kRun, 6, 7, -1, 0, 0));
    else if (i == 13)
      CPPUNIT_ASSERT(check(iterNum, kLumi, 6, 7, -1, 0, 0));
    else if (i == 14)
      CPPUNIT_ASSERT(check(iterNum, kLumi, 6, 8, 8, 0, 1));
    else if (i == 15)
      CPPUNIT_ASSERT(check(iterNum, kLumi, 6, 9, 8, 0, 1));
    else if (i == 16)
      CPPUNIT_ASSERT(check(iterNum, kEvent, 6, 9, 8, 0, 1));
    else
      CPPUNIT_ASSERT(false);

    switch (i) {
      case 0:
        CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == 3);
        break;
      case 10:
        CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == 3);
        break;
      case 12:
        CPPUNIT_ASSERT(iterNum.firstEventEntryThisRun() == 6);
        break;
    }
    switch (i) {
      case 0:
        CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == 3);
        break;
      case 1:
        CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == 3);
        break;
      case 2:
        CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == 3);
        break;
      case 3:
        CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == 3);
        break;
      case 10:
        CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == 5);
        break;
      case 12:
        CPPUNIT_ASSERT(iterNum.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
        break;
    }
  }
  CPPUNIT_ASSERT(i == 17);

  for (i = 0, iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder); iterNum != iterNumEnd; ++iterNum, ++i) {
    switch (i) {
      case 0:
        checkIDRunLumiEntry(iterNum, 0, 11, 0, 0);  // Run
        break;
      case 1:
        checkIDRunLumiEntry(iterNum, 0, 11, 101, 0);  // Lumi
        break;
      case 2:
        checkIDRunLumiEntry(iterNum, 0, 11, 101, 1);  // Lumi
        break;
      case 3:
        checkIDRunLumiEntry(iterNum, 0, 11, 101, 2);  // Lumi
        break;
      case 4:
        checkIDRunLumiEntry(iterNum, 0, 11, 101, 3);  // Event
        break;
      case 5:
        checkIDRunLumiEntry(iterNum, 0, 11, 101, 2);  // Event
        break;
      case 6:
        checkIDRunLumiEntry(iterNum, 0, 11, 101, 1);  // Event
        break;
      case 7:
        checkIDRunLumiEntry(iterNum, 0, 11, 101, 0);  // Event
        break;
      case 8:
        checkIDRunLumiEntry(iterNum, 0, 11, 102, 3);  // Lumi
        break;
      case 9:
        checkIDRunLumiEntry(iterNum, 0, 11, 102, 5);  // Event
        break;
      case 10:
        checkIDRunLumiEntry(iterNum, 0, 11, 102, 4);  // Event
        break;
      case 11:
        checkIDRunLumiEntry(iterNum, 1, 11, 0, 1);  // Run
        break;
      case 12:
        checkIDRunLumiEntry(iterNum, 1, 11, 0, 2);  // Run
        break;
      case 13:
        checkIDRunLumiEntry(iterNum, 1, 11, 101, 4);  // Lumi
        break;
      case 14:
        checkIDRunLumiEntry(iterNum, 1, 11, 102, 5);  // Lumi
        break;
      case 15:
        checkIDRunLumiEntry(iterNum, 1, 11, 102, 6);  // Lumi
        break;
      case 16:
        checkIDRunLumiEntry(iterNum, 1, 11, 102, 6);  // Event
        break;
    }
  }
  checkIDRunLumiEntry(iterNum, -1, 0, 0, -1);  // Event

  {
    edm::IndexIntoFile::IndexIntoFileItr iter3(
        &indexIntoFile, IndexIntoFile::entryOrder, IndexIntoFile::kEvent, 0, 3, 2, 1, 2);
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

    CPPUNIT_ASSERT(indexIntoFile.iterationWillBeInEntryOrder(IndexIntoFile::entryOrder) == true);

    edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterEntryCopy = iterEntry;
    edm::IndexIntoFile::IndexIntoFileItr iterEntryCopy2 = iterEntry;
    edm::IndexIntoFile::IndexIntoFileItr iterEntryEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
    int i = 0;
    for (i = 0; iterEntry != iterEntryEnd; ++iterEntry, ++iterEntryCopy, ++i) {
      CPPUNIT_ASSERT(iterEntry == iterEntryCopy);
      iterEntryCopy2 = iterEntryCopy;
      CPPUNIT_ASSERT(iterEntry == iterEntryCopy2);
      if (i == 0) {
        CPPUNIT_ASSERT(check(iterEntry, kRun, 0, 1, 1, 0, 2));
        CPPUNIT_ASSERT(iterEntry.indexIntoFile() == &indexIntoFile);
        CPPUNIT_ASSERT(iterEntry.size() == 10);
      } else if (i == 1)
        CPPUNIT_ASSERT(check(iterEntry, kLumi, 0, 1, 1, 0, 2));
      else if (i == 2)
        CPPUNIT_ASSERT(check(iterEntry, kLumi, 0, 2, 1, 0, 2));
      else if (i == 3)
        CPPUNIT_ASSERT(check(iterEntry, kLumi, 0, 3, 1, 0, 2));
      else if (i == 4)
        CPPUNIT_ASSERT(check(iterEntry, kEvent, 0, 3, 1, 0, 2));
      else if (i == 5)
        CPPUNIT_ASSERT(check(iterEntry, kEvent, 0, 3, 1, 1, 2));
      else if (i == 6)
        CPPUNIT_ASSERT(check(iterEntry, kEvent, 0, 3, 3, 0, 2));
      else if (i == 7)
        CPPUNIT_ASSERT(check(iterEntry, kEvent, 0, 3, 3, 1, 2));
      else if (i == 8)
        CPPUNIT_ASSERT(check(iterEntry, kLumi, 0, 4, 4, 0, 2));
      else if (i == 9)
        CPPUNIT_ASSERT(check(iterEntry, kEvent, 0, 4, 4, 0, 2));
      else if (i == 10)
        CPPUNIT_ASSERT(check(iterEntry, kEvent, 0, 4, 4, 1, 2));
      else if (i == 11)
        CPPUNIT_ASSERT(check(iterEntry, kRun, 5, 7, -1, 0, 0));
      else if (i == 12)
        CPPUNIT_ASSERT(check(iterEntry, kRun, 6, 7, -1, 0, 0));
      else if (i == 13)
        CPPUNIT_ASSERT(check(iterEntry, kLumi, 6, 7, -1, 0, 0));
      else if (i == 14)
        CPPUNIT_ASSERT(check(iterEntry, kLumi, 6, 8, 9, 0, 1));
      else if (i == 15)
        CPPUNIT_ASSERT(check(iterEntry, kLumi, 6, 9, 9, 0, 1));
      else if (i == 16)
        CPPUNIT_ASSERT(check(iterEntry, kEvent, 6, 9, 9, 0, 1));
      else
        CPPUNIT_ASSERT(false);

      switch (i) {
        case 0:
          CPPUNIT_ASSERT(iterEntry.firstEventEntryThisRun() == 0);
          break;
        case 10:
          CPPUNIT_ASSERT(iterEntry.firstEventEntryThisRun() == 0);
          break;
        case 12:
          CPPUNIT_ASSERT(iterEntry.firstEventEntryThisRun() == 6);
          break;
      }

      switch (i) {
        case 0:
          CPPUNIT_ASSERT(iterEntry.firstEventEntryThisLumi() == 0);
          break;
        case 1:
          CPPUNIT_ASSERT(iterEntry.firstEventEntryThisLumi() == 0);
          break;
        case 2:
          CPPUNIT_ASSERT(iterEntry.firstEventEntryThisLumi() == 0);
          break;
        case 3:
          CPPUNIT_ASSERT(iterEntry.firstEventEntryThisLumi() == 0);
          break;
        case 10:
          CPPUNIT_ASSERT(iterEntry.firstEventEntryThisLumi() == 4);
          break;
        case 12:
          CPPUNIT_ASSERT(iterEntry.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
          break;
      }
    }
    CPPUNIT_ASSERT(i == 17);

    for (i = 0, iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder); iterEntry != iterEntryEnd;
         ++iterEntry, ++i) {
      switch (i) {
        case 0:
          checkIDRunLumiEntry(iterEntry, 0, 11, 0, 0);  // Run
          break;
        case 1:
          checkIDRunLumiEntry(iterEntry, 0, 11, 101, 0);  // Lumi
          break;
        case 2:
          checkIDRunLumiEntry(iterEntry, 0, 11, 101, 1);  // Lumi
          break;
        case 3:
          checkIDRunLumiEntry(iterEntry, 0, 11, 101, 2);  // Lumi
          break;
        case 4:
          checkIDRunLumiEntry(iterEntry, 0, 11, 101, 0);  // Event
          break;
        case 5:
          checkIDRunLumiEntry(iterEntry, 0, 11, 101, 1);  // Event
          break;
        case 6:
          checkIDRunLumiEntry(iterEntry, 0, 11, 101, 2);  // Event
          break;
        case 7:
          checkIDRunLumiEntry(iterEntry, 0, 11, 101, 3);  // Event
          break;
        case 8:
          checkIDRunLumiEntry(iterEntry, 0, 11, 102, 3);  // Lumi
          break;
        case 9:
          checkIDRunLumiEntry(iterEntry, 0, 11, 102, 4);  // Event
          break;
        case 10:
          checkIDRunLumiEntry(iterEntry, 0, 11, 102, 5);  // Event
          break;
        case 11:
          checkIDRunLumiEntry(iterEntry, 1, 11, 0, 1);  // Run
          break;
        case 12:
          checkIDRunLumiEntry(iterEntry, 1, 11, 0, 2);  // Run
          break;
        case 13:
          checkIDRunLumiEntry(iterEntry, 1, 11, 101, 4);  // Lumi
          break;
        case 14:
          checkIDRunLumiEntry(iterEntry, 1, 11, 102, 5);  // Lumi
          break;
        case 15:
          checkIDRunLumiEntry(iterEntry, 1, 11, 102, 6);  // Lumi
          break;
        case 16:
          checkIDRunLumiEntry(iterEntry, 1, 11, 102, 6);  // Event
          break;
      }
    }
    checkIDRunLumiEntry(iterEntry, -1, 0, 0, -1);  // Event
  }
  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 1, 0, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 1, 1, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 1);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 3, 0, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 2);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 3, 1, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 4, 4, 0, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 4, 4, 1, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));

  skipEventForward(iterFirst);
  checkSkipped(1, 11, 102, 6);
  CPPUNIT_ASSERT(check(iterFirst, kEnd, -1, -1, -1, 0, 0));

  skipEventForward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  CPPUNIT_ASSERT(check(iterFirst, kEnd, -1, -1, -1, 0, 0));

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 1, 1, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 1);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 3, 0, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 2);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 3, 1, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 0, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 1, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));

  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  skipEventForward(iterFirst);
  checkSkipped(1, 11, 102, 6);
  CPPUNIT_ASSERT(check(iterFirst, kEnd, -1, -1, -1, 0, 0));

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 1, 1, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 1);
  CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 3, 0, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 2);
  CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 3, 1, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 0, 2));

  ++iterFirst;
  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 1, 2));

  skipEventForward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  iterFirst.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 1, 0, 2));
  iterFirst.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 0, 2));
  iterFirst.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));
  iterFirst.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 7, -1, 0, 0));
  iterFirst.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 8, 9, 0, 1));
  iterFirst.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterFirst, kEnd, -1, -1, -1, 0, 0));

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 1, 0, 2));
  iterFirst.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 0, 2));
  ++iterFirst;
  iterFirst.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));

  iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 1, 0, 2));
  iterFirst.advanceToNextRun();
  CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));

  // Repeat skip tests with the other sort order

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 0, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 1, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 2);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 2, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 1);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 3, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 4, 4, 0, 2));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 4, 4, 1, 2));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterNum, kRun, 5, 7, -1, 0, 0));

  skipEventForward(iterNum);
  checkSkipped(1, 11, 102, 6);
  CPPUNIT_ASSERT(check(iterNum, kEnd, -1, -1, -1, 0, 0));

  skipEventForward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  CPPUNIT_ASSERT(check(iterNum, kEnd, -1, -1, -1, 0, 0));

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 1, 1, 1, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 2);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 1, 1, 2, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 1);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 1, 1, 3, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 4, 4, 0, 2));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 4, 4, 1, 2));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterNum, kRun, 5, 7, -1, 0, 0));

  ++iterNum;
  ++iterNum;
  ++iterNum;
  CPPUNIT_ASSERT(check(iterNum, kLumi, 6, 8, 8, 0, 1));
  skipEventForward(iterNum);
  checkSkipped(1, 11, 102, 6);
  CPPUNIT_ASSERT(check(iterNum, kEnd, -1, -1, -1, 0, 0));

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 3, 1, 1, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 2);
  CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 3, 1, 2, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 1);
  CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 3, 1, 3, 4));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 4, 4, 0, 2));

  ++iterNum;
  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 4, 4, 1, 2));

  skipEventForward(iterNum);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterNum, kRun, 5, 7, -1, 0, 0));

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  iterNum.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 1, 1, 0, 4));
  iterNum.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 4, 4, 0, 2));
  iterNum.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterNum, kRun, 5, 7, -1, 0, 0));
  iterNum.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterNum, kLumi, 6, 7, -1, 0, 0));
  iterNum.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterNum, kLumi, 6, 8, 8, 0, 1));
  iterNum.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterNum, kEnd, -1, -1, -1, 0, 0));

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 3, 1, 0, 4));
  iterNum.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 4, 4, 0, 2));
  ++iterNum;
  iterNum.advanceToNextLumiOrRun();
  CPPUNIT_ASSERT(check(iterNum, kRun, 5, 7, -1, 0, 0));

  iterNum = indexIntoFile.begin(IndexIntoFile::numericalOrder);
  ++iterNum;
  ++iterNum;
  ++iterNum;
  ++iterNum;
  CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 3, 1, 0, 4));
  iterNum.advanceToNextRun();
  CPPUNIT_ASSERT(check(iterNum, kRun, 5, 7, -1, 0, 0));

  // Check backwards iteration

  iterFirst = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);

  skipEventBackward(iterFirst);
  checkSkipped(1, 11, 102, 6);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 8, 9, 0, 1));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 4, 4, 1, 2));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 4, 4, 0, 2));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 3, 1, 2));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 2);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 3, 0, 2));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 1);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 1, 1, 2));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 1, 0, 2));

  skipEventBackward(iterFirst);
  checkSkipped(-1, 0, 0, -1);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 1, 0, 2));

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
  CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 1, 2));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 0, 2));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 3, 1, 2));

  skipEventForward(iterFirst);
  skipEventForward(iterFirst);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 1, 2));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 0, 2));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 3, 1, 2));

  iterFirst.advanceToNextRun();
  iterFirst.advanceToEvent();
  CPPUNIT_ASSERT(check(iterFirst, kEvent, 6, 9, 9, 0, 1));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 4, 4, 1, 2));

  iterFirst.advanceToNextRun();
  ++iterFirst;
  ++iterFirst;
  ++iterFirst;
  CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 8, 9, 0, 1));

  skipEventBackward(iterFirst);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 4, 4, 1, 2));

  iterNum = indexIntoFile.end(IndexIntoFile::numericalOrder);

  skipEventBackward(iterNum);
  checkSkipped(1, 11, 102, 6);
  CPPUNIT_ASSERT(check(iterNum, kRun, 5, 8, 8, 0, 1));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 4, 4, 1, 2));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 4, 4, 0, 2));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 3, 4));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 1);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 2, 4));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 2);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 1, 4));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 3);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 0, 4));

  skipEventBackward(iterNum);
  checkSkipped(-1, 0, 0, -1);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 1, 1, 0, 4));

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
  CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 4, 4, 1, 2));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterNum, kEvent, 0, 4, 4, 0, 2));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 1, 1, 3, 4));

  skipEventForward(iterNum);
  skipEventForward(iterNum);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 4, 4, 1, 2));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 5);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 4, 4, 0, 2));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 101, 0);
  CPPUNIT_ASSERT(check(iterNum, kLumi, 0, 1, 1, 3, 4));

  iterNum.advanceToNextRun();
  iterNum.advanceToEvent();
  CPPUNIT_ASSERT(check(iterNum, kEvent, 6, 9, 8, 0, 1));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 4, 4, 1, 2));

  iterNum.advanceToNextRun();
  ++iterNum;
  ++iterNum;
  ++iterNum;
  CPPUNIT_ASSERT(check(iterNum, kLumi, 6, 8, 8, 0, 1));

  skipEventBackward(iterNum);
  checkSkipped(0, 11, 102, 4);
  CPPUNIT_ASSERT(check(iterNum, kRun, 0, 4, 4, 1, 2));
}

void TestIndexIntoFile3::testOverlappingLumis() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 11, 104, 0, 0);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 103, 7, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 103, 6, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 103, 5, 2);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 103, 4, 3);  // Event
  //Dummy Lumi gets added
  indexIntoFile.addEntry(fakePHID1, 11, 102, 5, 4);  // Event
  //Another dummy lumi gets added
  indexIntoFile.addEntry(fakePHID1, 11, 103, 0, 1);  // Lumi

  indexIntoFile.addEntry(fakePHID1, 11, 102, 4, 5);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 102, 0, 2);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 0);    // Run
  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 1);    // Run
  indexIntoFile.addEntry(fakePHID2, 11, 101, 0, 3);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 4);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 4, 6);  // Event
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 5);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 2);    // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  std::vector<IndexIntoFile::EventEntry>& eventEntries = indexIntoFile.eventEntries();
  eventEntries.emplace_back(5, 4);
  eventEntries.emplace_back(4, 5);
  eventEntries.emplace_back(7, 0);
  eventEntries.emplace_back(6, 1);
  eventEntries.emplace_back(5, 2);
  eventEntries.emplace_back(4, 3);
  eventEntries.emplace_back(4, 6);
  indexIntoFile.sortEventEntries();

  std::vector<LuminosityBlockNumber_t> lumis;

  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      if (i == 0) {
        CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, -1, 0, 0));
        CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
        CPPUNIT_ASSERT(iterFirst.size() == 11);
      }
      //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
      else if (i == 1)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, -1, 0, 0));
      else if (i == 2)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 3, 2, 0, 4));
      else if (i == 3)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 2, 0, 4));
      else if (i == 4)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 2, 1, 4));
      else if (i == 5)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 2, 2, 4));
      else if (i == 6)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 2, 3, 4));
      else if (i == 7)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 5, 4, 0, 1));
      else if (i == 8)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 5, 4, 0, 1));
      else if (i == 9)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 5, 5, 0, 1));
      else if (i == 10)
        CPPUNIT_ASSERT(check(iterFirst, kRun, 6, 8, -1, 0, 0));
      else if (i == 11)
        CPPUNIT_ASSERT(check(iterFirst, kRun, 7, 8, -1, 0, 0));
      else if (i == 12)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 7, 8, -1, 0, 0));
      else if (i == 13)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 7, 9, 10, 0, 1));
      else if (i == 14)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 7, 10, 10, 0, 1));
      else if (i == 15)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 7, 10, 10, 0, 1));
      else
        CPPUNIT_ASSERT(false);

      switch (i) {
        case 1:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          break;
        case 9:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          break;
        case 11:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 6);
          break;
      }

      switch (i) {
        case 2:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        case 9:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 4);
          break;
        case 11:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
          break;
      }
    }
    CPPUNIT_ASSERT(i == 16);
  }
  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::numericalOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::numericalOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
      if (i == 0) {
        CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 2, 1, 0, 2));

        iterFirst.getLumisInRun(lumis);
        std::vector<LuminosityBlockNumber_t> expected{102, 103, 104};
        CPPUNIT_ASSERT(lumis == expected);
      } else if (i == 1)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 2, 1, 0, 2));
      else if (i == 2)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 0, 2));
      else if (i == 3)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 1, 2));
      else if (i == 4)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 3, 0, 4));
      else if (i == 5)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 0, 4));
      else if (i == 6)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 1, 4));
      else if (i == 7)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 2, 4));
      else if (i == 8)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 3, 4));
      else if (i == 9)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 5, -1, 0, 0));
      else if (i == 10)
        CPPUNIT_ASSERT(check(iterFirst, kRun, 6, 8, -1, 0, 0));
      else if (i == 11)
        CPPUNIT_ASSERT(check(iterFirst, kRun, 7, 8, -1, 0, 0));
      else if (i == 12)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 7, 8, -1, 0, 0));
      else if (i == 13)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 7, 9, 9, 0, 1));
      else if (i == 14)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 7, 10, 9, 0, 1));
      else if (i == 15)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 7, 10, 9, 0, 1));
      else
        CPPUNIT_ASSERT(false);
    }
    CPPUNIT_ASSERT(i == 16);
  }
  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::entryOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
      if (i == 0) {
        CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, -1, 0, 0));
        CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
        CPPUNIT_ASSERT(iterFirst.size() == 11);
      }
      //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
      else if (i == 1)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, -1, 0, 0));
      else if (i == 2)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 3, 2, 0, 4));
      else if (i == 3)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 2, 0, 4));
      else if (i == 4)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 2, 1, 4));
      else if (i == 5)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 2, 2, 4));
      else if (i == 6)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 2, 3, 4));
      else if (i == 7)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 5, 4, 0, 1));
      else if (i == 8)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 5, 4, 0, 1));
      else if (i == 9)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 5, 5, 0, 1));
      else if (i == 10)
        CPPUNIT_ASSERT(check(iterFirst, kRun, 6, 8, -1, 0, 0));
      else if (i == 11)
        CPPUNIT_ASSERT(check(iterFirst, kRun, 7, 8, -1, 0, 0));
      else if (i == 12)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 7, 8, -1, 0, 0));
      else if (i == 13)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 7, 9, 10, 0, 1));
      else if (i == 14)
        CPPUNIT_ASSERT(check(iterFirst, kLumi, 7, 10, 10, 0, 1));
      else if (i == 15)
        CPPUNIT_ASSERT(check(iterFirst, kEvent, 7, 10, 10, 0, 1));
      else
        CPPUNIT_ASSERT(false);

      switch (i) {
        case 1:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          break;
        case 9:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          break;
        case 11:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 6);
          break;
      }

      switch (i) {
        case 2:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        case 9:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 4);
          break;
        case 11:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
          break;
      }
    }
    CPPUNIT_ASSERT(i == 16);
  }

  {
    edm::IndexIntoFile::IndexIntoFileItr testIter = indexIntoFile.findPosition(11, 103, 7);
    CPPUNIT_ASSERT(check(testIter, kRun, 0, 4, 3, 3, 4));
    ++testIter;
    CPPUNIT_ASSERT(check(testIter, kLumi, 0, 4, 3, 3, 4));
    ++testIter;
    CPPUNIT_ASSERT(check(testIter, kEvent, 0, 4, 3, 3, 4));
    ++testIter;
    CPPUNIT_ASSERT(check(testIter, kLumi, 0, 5, -1, 0, 0));
  }
  {
    edm::IndexIntoFile::IndexIntoFileItr testIter = indexIntoFile.findPosition(11, 0, 7);
    CPPUNIT_ASSERT(check(testIter, kRun, 0, 4, 3, 3, 4));
    ++testIter;
    CPPUNIT_ASSERT(check(testIter, kLumi, 0, 4, 3, 3, 4));
    ++testIter;
    CPPUNIT_ASSERT(check(testIter, kEvent, 0, 4, 3, 3, 4));
    ++testIter;
    CPPUNIT_ASSERT(check(testIter, kLumi, 0, 5, -1, 0, 0));
    skipEventBackward(testIter);
    CPPUNIT_ASSERT(check(testIter, kLumi, 0, 4, 3, 3, 4));
  }
}

void TestIndexIntoFile3::testOverlappingLumisMore() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 11, 101, 7, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 6, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 5, 2);  // Event
  //Dummy Lumi gets added
  indexIntoFile.addEntry(fakePHID1, 11, 102, 5, 4);  // Event
  //Another dummy lumi gets added
  indexIntoFile.addEntry(fakePHID1, 11, 101, 4, 3);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 0);  // Lumi

  indexIntoFile.addEntry(fakePHID1, 11, 102, 4, 5);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 102, 0, 1);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 0);    // Run
  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 1);    // Run
  indexIntoFile.addEntry(fakePHID2, 11, 101, 0, 2);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 3);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 4, 6);  // Event
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 4);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 2);    // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  std::vector<LuminosityBlockNumber_t> lumis;

  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 2, 1, 0, 3));
          CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
          CPPUNIT_ASSERT(iterFirst.size() == 10);

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{101, 102};
          CPPUNIT_ASSERT(lumis == expected);
          break;
        }
          //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
        case 1: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 2, 1, 0, 3));
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 0, 3));
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 1, 3));
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 2, 3));
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 2, 0, 1));
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 3, 0, 1));
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 0, 1));
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 0, 1));
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{101, 102};
          CPPUNIT_ASSERT(lumis == expected);
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 6, 7, -1, 0, 0));
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 7, -1, 0, 0));
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 8, 9, 0, 1));
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 9, 9, 0, 1));
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 6, 9, 9, 0, 1));
          break;
        }
        default: {
          CPPUNIT_ASSERT(false);
        }
      }

      switch (i) {
        case 0:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          break;
        case 8:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          break;
        case 10:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 6);
          break;
      }

      switch (i) {
        case 0:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        case 8:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 4);
          break;
        case 10:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
          break;
      }
    }
    CPPUNIT_ASSERT(i == 15);
  }

  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::entryOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 2, 1, 0, 3));
          CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
          CPPUNIT_ASSERT(iterFirst.size() == 10);

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{101, 102};
          CPPUNIT_ASSERT(lumis == expected);
          break;
        }
        //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
        case 1: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 2, 1, 0, 3));
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 0, 3));
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 1, 3));
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 2, 3));
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 2, 0, 1));
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 3, 0, 1));
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 0, 1));
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 0, 1));
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{101, 102};
          CPPUNIT_ASSERT(lumis == expected);
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 6, 7, -1, 0, 0));
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 7, -1, 0, 0));
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 8, 9, 0, 1));
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 9, 9, 0, 1));
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 6, 9, 9, 0, 1));
          break;
        }
        default:
          CPPUNIT_ASSERT(false);
      }

      switch (i) {
        case 0:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          break;
        case 8:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          break;
        case 10:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 6);
          break;
      }

      switch (i) {
        case 0:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        case 8:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 4);
          break;
        case 10:
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
          break;
      }
    }
    CPPUNIT_ASSERT(i == 15);
  }
}

void TestIndexIntoFile3::testOverlappingLumisOutOfOrderEvent() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 11, 101, 7, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 6, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 5, 2);  // Event
  // This test is a bit misnamed. The difference between this test and
  // the preceding one lies in the event entry numbers being swapped
  // in order. The order below is the expected order (it is NOT out of order).
  // I don't think the order of entry numbers in the preceding test is
  // possible, the event entry number is incremented as each event
  // is written and addEntry is called at that time. The event entry
  // number should always increment by one at each addEntry call for
  // an event. Possibly the preceding test should be deleted (although
  // IndexIntoFile works with that order also even if it will never
  // occur ...)
  //Dummy Lumi gets added
  indexIntoFile.addEntry(fakePHID1, 11, 102, 5, 3);  // Event
  //Another dummy lumi gets added
  indexIntoFile.addEntry(fakePHID1, 11, 101, 4, 4);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 101, 0, 0);  // Lumi

  indexIntoFile.addEntry(fakePHID1, 11, 102, 4, 5);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 102, 0, 1);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 0);    // Run
  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 1);    // Run
  indexIntoFile.addEntry(fakePHID2, 11, 101, 0, 2);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 3);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 102, 4, 6);  // Event
  indexIntoFile.addEntry(fakePHID2, 11, 102, 0, 4);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 2);    // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  std::vector<LuminosityBlockNumber_t> lumis;

  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 2, 1, 0, 3));  // run 11
          CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
          CPPUNIT_ASSERT(iterFirst.size() == 10);

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{101, 102};
          CPPUNIT_ASSERT(lumis == expected);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        }
          //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
        case 1: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 2, 1, 0, 3));  // lumi 11/101
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 0, 3));  // event 11/101/7
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 1, 3));  // event 11/101/6
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 1, 2, 3));  // event 11/101/5
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 2, 0, 1));
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 3, 0, 1));
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 0, 1));
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 0, 1));
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 3);
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{101, 102};
          CPPUNIT_ASSERT(lumis == expected);
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 6, 7, -1, 0, 0));
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 6);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 7, -1, 0, 0));
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 8, 9, 0, 1));
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 9, 9, 0, 1));
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 6, 9, 9, 0, 1));
          break;
        }
        default: {
          CPPUNIT_ASSERT(false);
        }
      }
    }
    CPPUNIT_ASSERT(i == 15);
  }

  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::entryOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      /*std::cout << "out of order run:" << iterFirst.run() << " lumi:" << iterFirst.lumi()
                << " firstEventEntryThisRun:" << iterFirst.firstEventEntryThisRun()
                << " firstEventEntryThisLumi:" << iterFirst.firstEventEntryThisLumi() << std::endl; */
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 1, 0, 3));  // run 11
          CPPUNIT_ASSERT(iterFirst.processHistoryIDIndex() == 0);
          CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
          CPPUNIT_ASSERT(iterFirst.size() == 10);
          CPPUNIT_ASSERT(iterFirst.shouldProcessRun());

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{101, 102};
          CPPUNIT_ASSERT(lumis == expected);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        }
        //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
        case 1: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 1, 0, 3));  // lumi 11/101
          CPPUNIT_ASSERT(iterFirst.processHistoryIDIndex() == 0);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 1, 1, 0, 3));  // event 11/101/7
          CPPUNIT_ASSERT(iterFirst.processHistoryIDIndex() == 0);
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 1, 1, 1, 3));  // event 11/101/6
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 1, 1, 2, 3));  // event 11/101/5
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 2, 2, 0, 1));  // lumi 11/102
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 2, 0, 1));  // event 11/102/3
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 3, 3, 0, 1));  // lumi 11/101
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 3, 0, 1));  // event 11/101/4
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 4);
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 0, 1));  // lumi 11/102
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 0, 1));  // event 11/102/4
          CPPUNIT_ASSERT(iterFirst.processHistoryIDIndex() == 0);
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 5, 7, -1, 0, 0));  //Run Phid 2 11
          CPPUNIT_ASSERT(iterFirst.shouldProcessRun());

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{101, 102};
          CPPUNIT_ASSERT(lumis == expected);

          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 6);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == IndexIntoFile::invalidIndex);
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 6, 7, -1, 0, 0));
          CPPUNIT_ASSERT(iterFirst.processHistoryIDIndex() == 1);
          CPPUNIT_ASSERT(iterFirst.shouldProcessRun());
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 7, -1, 0, 0));
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 8, 9, 0, 1));
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 15: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 6, 9, 9, 0, 1));
          CPPUNIT_ASSERT(iterFirst.processHistoryIDIndex() == 1);
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 16: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 6, 9, 9, 0, 1));
          CPPUNIT_ASSERT(iterFirst.processHistoryIDIndex() == 1);
          break;
        }
        default:
          CPPUNIT_ASSERT(false);
      }
    }
    CPPUNIT_ASSERT(i == 17);
  }
}

void TestIndexIntoFile3::testOverlappingLumisWithEndWithEmptyLumi() {
  // from a failed job
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 1, 2, 0);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 5, 1);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 8, 2);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 6, 3);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 3, 4);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 9, 5);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 4, 6);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 1, 7);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 7, 8);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 0);    // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 2, 10, 9);   // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 1);    // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 3, 12, 10);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 3, 11, 11);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 3, 15, 12);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 3, 14, 13);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 18, 14);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 17, 15);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 20, 16);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 3, 13, 17);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 3, 0, 2);    // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 4, 19, 18);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 16, 19);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 0, 3);    // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);    // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  std::vector<LuminosityBlockNumber_t> lumis;

  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      //std::cout << "out of order run:" << iterFirst.run() << " lumi:" << iterFirst.lumi()
      //          << " firstEventEntryThisRun:" << iterFirst.firstEventEntryThisRun()
      //          << " firstEventEntryThisLumi:" << iterFirst.firstEventEntryThisLumi() << std::endl;
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 4, 1, 0, 2));  // run 1
          CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
          CPPUNIT_ASSERT(iterFirst.size() == 13);
          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{1, 2, 3, 4};
          CPPUNIT_ASSERT(lumis == expected);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        }
          //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
        case 1: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 1, 0, 2));  // lumi 1/1
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 1, 0, 2));  // event 1/1/2
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 1, 1, 2));  // event 1/1/5
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 2, 0, 1));  // event 1/1/3
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 0, 2));  // event 1/1/4
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 1, 2));  // event 1/1/1
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 8, 5, 0, 2));  // lumi 1/2
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 5, 0, 2));  // event 1/2/8
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 2);
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 5, 1, 2));  // event 1/2/6
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 6, 0, 1));  // event 1/2/9
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 7, 0, 1));  // event 1/2/7
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 8, 0, 1));  // event 1/2/10
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 10, 9, 0, 4));  // lumi 1/3
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 9, 0, 4));  // event 1/3/12
          break;
        }
        case 15: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 9, 1, 4));  // event 1/3/11
          break;
        }
        case 16: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 9, 2, 4));  // event 1/3/15
          break;
        }
        case 17: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 9, 3, 4));  // event 1/3/14
          break;
        }
        case 18: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 0, 1));  // event 1/3/13
          break;
        }
        case 19: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 12, 11, 0, 3));  // lumi 1/4
          break;
        }
        case 20: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 12, 11, 0, 3));  // event 1/4/18
          break;
        }
        case 21: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 12, 11, 1, 3));  // event 1/4/17
          break;
        }
        case 22: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 12, 11, 2, 3));  // event 1/4/20
          break;
        }
        case 23: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 12, 12, 0, 2));  // event 1/4/19
          break;
        }
        case 24: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 12, 12, 1, 2));  // event 1/4/16
          break;
        }
        default: {
          CPPUNIT_ASSERT(false);
        }
      }
    }
    CPPUNIT_ASSERT(i == 25);
  }

  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::entryOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      //std::cout << "out of order run:" << iterFirst.run() << " lumi:" << iterFirst.lumi()
      //          << " firstEventEntryThisRun:" << iterFirst.firstEventEntryThisRun()
      //          << " firstEventEntryThisLumi:" << iterFirst.firstEventEntryThisLumi() << std::endl;
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 1, 0, 2));  // run 1
          CPPUNIT_ASSERT(iterFirst.run() == 1);
          CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
          CPPUNIT_ASSERT(iterFirst.size() == 13);
          CPPUNIT_ASSERT(iterFirst.shouldProcessRun());

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{1, 2, 3, 4};
          //CPPUNIT_ASSERT(lumis == expected);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        }
        //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
        case 1: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 1, 0, 2));  // lumi 1/1
          CPPUNIT_ASSERT(iterFirst.lumi() == 1);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 1, 1, 0, 2));  // event 1/1/2
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 1, 1, 1, 2));  // event 1/1/5
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 2, 2, 0, 2));  // lumi 1/2
          CPPUNIT_ASSERT(iterFirst.lumi() == 2);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 2, 0, 2));  // event 1/2/8
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 2, 1, 2));  // event 1/2/6
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 3, 3, 0, 1));  // lumi 1/1
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          CPPUNIT_ASSERT(iterFirst.lumi() == 1);
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 3, 0, 1));  // event 1/1/3
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 4);
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 0, 1));  // lumi 1/2
          CPPUNIT_ASSERT(iterFirst.lumi() == 2);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 0, 1));  // event 1/2/9
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 6, 5, 0, 2));  // lumi 1/1   5
          CPPUNIT_ASSERT(iterFirst.lumi() == 1);
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 6, 5, 0, 2));  // event 1/1/4
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 6, 5, 1, 2));  // event 1/1/1
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 8, 7, 0, 1));  // lumi 1/2
          CPPUNIT_ASSERT(iterFirst.lumi() == 2);
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 15: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 7, 0, 1));  // event 1/2/7
          break;
        }
        case 16: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 8, 0, 1));  // event 1/2/10
          CPPUNIT_ASSERT(iterFirst.lumi() == 2);
          break;
        }
        case 17: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 9, 9, 0, 4));  // lumi 1/3
          CPPUNIT_ASSERT(iterFirst.lumi() == 3);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 18: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 0, 4));  // event 1/3/12
          break;
        }
        case 19: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 1, 4));  // event 1/3/11
          break;
        }
        case 20: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 2, 4));  // event 1/3/15
          break;
        }
        case 21: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 3, 4));  // event 1/3/14
          break;
        }
        case 22: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 10, 10, 0, 3));  // lumi 1/4
          CPPUNIT_ASSERT(iterFirst.lumi() == 4);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 23: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 0, 3));  // event 1/4/16
          break;
        }
        case 24: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 1, 3));  // event 1/4/17
          break;
        }
        case 25: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 2, 3));  // event 1/4/18
          break;
        }
        case 26: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 11, 11, 0, 1));  // lumi 1/3
          CPPUNIT_ASSERT(iterFirst.lumi() == 3);
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 27: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 11, 11, 0, 1));  // event 1/3/13
          break;
        }
        case 28: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 12, 12, 0, 2));  // lumi 1/4
          CPPUNIT_ASSERT(iterFirst.lumi() == 4);
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 29: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 12, 12, 0, 2));  // event 1/4/19
          break;
        }
        case 30: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 12, 12, 1, 2));  // event 1/4/16
          break;
        }

        default:
          CPPUNIT_ASSERT(false);
      }
    }
    CPPUNIT_ASSERT(i == 31);
  }
}

void TestIndexIntoFile3::testOverlappingLumisWithLumiEndOrderChanged() {
  // from a failed job
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 1, 2, 0);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 5, 1);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 8, 2);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 6, 3);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 3, 4);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 9, 5);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 4, 6);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 7, 7);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 10, 8);   // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 0);    // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 1, 1, 9);    // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 1);    // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 3, 12, 10);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 3, 11, 11);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 3, 15, 12);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 3, 14, 13);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 18, 14);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 17, 15);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 20, 16);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 19, 17);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 16, 18);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 4, 0, 2);    // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 3, 13, 19);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 3, 0, 3);    // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);    // Run
  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  std::vector<LuminosityBlockNumber_t> lumis;

  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      //std::cout << "out of order run:" << iterFirst.run() << " lumi:" << iterFirst.lumi()
      //          << " firstEventEntryThisRun:" << iterFirst.firstEventEntryThisRun()
      //          << " firstEventEntryThisLumi:" << iterFirst.firstEventEntryThisLumi() << std::endl;
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 4, 1, 0, 2));  // run 1
          CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
          CPPUNIT_ASSERT(iterFirst.size() == 11);
          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{1, 2, 3, 4};
          CPPUNIT_ASSERT(lumis == expected);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        }
          //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
        case 1: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 1, 0, 2));  // lumi 1/1
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 1, 0, 2));  // event 1/1/2
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 1, 1, 2));  // event 1/1/5
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 2, 0, 1));  // event 1/1/3
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 3, 0, 1));  // event 1/1/4
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 0, 1));  // event 1/1/1
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 7, 5, 0, 2));  // lumi 1/2
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 7, 5, 0, 2));  // event 1/2/8
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 2);
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 7, 5, 1, 2));  // event 1/2/6
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 7, 6, 0, 1));  // event 1/2/9
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 7, 7, 0, 2));  // event 1/2/7
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 7, 7, 1, 2));  // event 1/2/10
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 9, 8, 0, 4));  // lumi 1/3
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 8, 0, 4));  // event 1/3/12
          break;
        }
        case 15: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 8, 1, 4));  // event 1/3/11
          break;
        }
        case 16: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 8, 2, 4));  // event 1/3/15
          break;
        }
        case 17: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 8, 3, 4));  // event 1/3/14
          break;
        }
        case 18: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 0, 1));  // event 1/3/13
          break;
        }
        case 19: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 10, 10, 0, 5));  // lumi 1/4
          break;
        }
        case 20: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 0, 5));  // event 1/4/18
          break;
        }
        case 21: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 1, 5));  // event 1/4/17
          break;
        }
        case 22: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 2, 5));  // event 1/4/20
          break;
        }
        case 23: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 3, 5));  // event 1/4/19
          break;
        }
        case 24: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 4, 5));  // event 1/4/16
          break;
        }
        default: {
          CPPUNIT_ASSERT(false);
        }
      }
    }
    CPPUNIT_ASSERT(i == 25);
  }

  {
    edm::IndexIntoFile::IndexIntoFileItr iterFirst = indexIntoFile.begin(IndexIntoFile::entryOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterFirstEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
    int i = 0;
    for (i = 0; iterFirst != iterFirstEnd; ++iterFirst, ++i) {
      //std::cout << "out of order run:" << iterFirst.run() << " lumi:" << iterFirst.lumi()
      //          << " firstEventEntryThisRun:" << iterFirst.firstEventEntryThisRun()
      //          << " firstEventEntryThisLumi:" << iterFirst.firstEventEntryThisLumi() << std::endl;
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iterFirst, kRun, 0, 1, 1, 0, 2));  // run 1
          CPPUNIT_ASSERT(iterFirst.shouldProcessRun());
          CPPUNIT_ASSERT(iterFirst.run() == 1);
          CPPUNIT_ASSERT(iterFirst.indexIntoFile() == &indexIntoFile);
          CPPUNIT_ASSERT(iterFirst.size() == 11);

          iterFirst.getLumisInRun(lumis);
          std::vector<LuminosityBlockNumber_t> expected{1, 2, 3, 4};
          //CPPUNIT_ASSERT(lumis == expected);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 0);
          break;
        }
        //values are 'IndexIntoFile::EntryType' 'indexToRun' 'indexToLumi' 'indexToEventRange' 'indexToEvent' 'nEvents'
        case 1: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 1, 1, 0, 2));  // lumi 1/1
          CPPUNIT_ASSERT(iterFirst.lumi() == 1);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 1, 1, 0, 2));  // event 1/1/2
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 1, 1, 1, 2));  // event 1/1/5
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 2, 2, 0, 2));  // lumi 1/2
          CPPUNIT_ASSERT(iterFirst.lumi() == 2);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 2, 0, 2));  // event 1/2/8
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 2, 2, 1, 2));  // event 1/2/6
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 3, 3, 0, 1));  // lumi 1/1
          CPPUNIT_ASSERT(iterFirst.lumi() == 1);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 3, 3, 0, 1));  // event 1/1/3
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisRun() == 0);
          CPPUNIT_ASSERT(iterFirst.firstEventEntryThisLumi() == 4);
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 4, 4, 0, 1));  // lumi 1/2
          CPPUNIT_ASSERT(iterFirst.lumi() == 2);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 4, 4, 0, 1));  // event 1/2/9
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 5, 5, 0, 1));  // lumi 1/1   5
          CPPUNIT_ASSERT(iterFirst.lumi() == 1);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 5, 5, 0, 1));  // event 1/1/4
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 6, 6, 0, 2));  // lumi 1/2
          CPPUNIT_ASSERT(iterFirst.lumi() == 2);
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 6, 6, 0, 2));  // event 1/2/7
          break;
        }
        case 15: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 6, 6, 1, 2));  // event 1/2/10
          CPPUNIT_ASSERT(iterFirst.lumi() == 2);
          break;
        }
        case 16: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 7, 7, 0, 1));  // lumi 1/1
          CPPUNIT_ASSERT(iterFirst.lumi() == 1);
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 17: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 7, 7, 0, 1));  // event 1/1/1
          break;
        }
        case 18: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 8, 8, 0, 4));  // lumi 1/3
          CPPUNIT_ASSERT(iterFirst.lumi() == 3);
          CPPUNIT_ASSERT(!iterFirst.shouldProcessLumi());
          break;
        }
        case 19: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 8, 0, 4));  // event 1/3/12
          break;
        }
        case 20: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 8, 1, 4));  // event 1/3/11
          break;
        }
        case 21: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 8, 2, 4));  // event 1/3/15
          break;
        }
        case 22: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 8, 8, 3, 4));  // event 1/3/14
          break;
        }
        case 23: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 9, 9, 0, 5));  // lumi 1/4
          CPPUNIT_ASSERT(iterFirst.lumi() == 4);
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 24: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 0, 5));  // event 1/4/16
          break;
        }
        case 25: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 1, 5));  // event 1/4/17
          break;
        }
        case 26: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 2, 5));  // event 1/4/18
          break;
        }
        case 27: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 3, 5));  // event 1/4/19
          break;
        }
        case 28: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 9, 9, 4, 5));  // event 1/4/16
          break;
        }
        case 29: {
          CPPUNIT_ASSERT(check(iterFirst, kLumi, 0, 10, 10, 0, 1));  // lumi 1/3
          CPPUNIT_ASSERT(iterFirst.lumi() == 3);
          CPPUNIT_ASSERT(iterFirst.shouldProcessLumi());
          break;
        }
        case 30: {
          CPPUNIT_ASSERT(check(iterFirst, kEvent, 0, 10, 10, 0, 1));  // event 1/3/13
          break;
        }

        default:
          CPPUNIT_ASSERT(false);
      }
    }
    CPPUNIT_ASSERT(i == 31);
  }
}

void TestIndexIntoFile3::testNonContiguousRun() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 1, 1, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 2, 1, 1, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 2, 2);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 0);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 0);  // Run
  indexIntoFile.addEntry(fakePHID1, 2, 1, 0, 1);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 1);  // Run

  indexIntoFile.addEntry(fakePHID1, 1, 1, 3, 3);  // Event
  indexIntoFile.addEntry(fakePHID1, 2, 1, 2, 4);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 4, 5);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 2);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 2);  // Run
  indexIntoFile.addEntry(fakePHID1, 2, 1, 0, 3);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 3);  // Run

  indexIntoFile.sortVector_Run_Or_Lumi_Entries();
  {
    edm::IndexIntoFile::IndexIntoFileItr iter = indexIntoFile.begin(IndexIntoFile::entryOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
    int i = 0;
    for (i = 0; iter != iterEnd; ++iter, ++i) {
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iter, kRun, 0, 1, 1, 0, 1));  // run 1
          CPPUNIT_ASSERT(iter.size() == 12);
          CPPUNIT_ASSERT(iter.indexedSize() == 16);
          CPPUNIT_ASSERT(!iter.shouldProcessRun());
          CPPUNIT_ASSERT(iter.entry() == 0);
          break;
        }
        case 1: {
          CPPUNIT_ASSERT(check(iter, kLumi, 0, 1, 1, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(!iter.shouldProcessLumi());
          // entry has special code in it to look for other entries if the
          // current RunOrLumiEntry has an invalid TTree entry number.
          CPPUNIT_ASSERT(iter.entry() == 0);
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iter, kEvent, 0, 1, 1, 0, 1));  // event 1:1:1
          CPPUNIT_ASSERT(iter.entry() == 0);
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iter, kRun, 2, 3, 3, 0, 1));  // run 2
          CPPUNIT_ASSERT(!iter.shouldProcessRun());
          CPPUNIT_ASSERT(iter.entry() == 1);
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iter, kLumi, 2, 3, 3, 0, 1));  // lumi 2:1
          CPPUNIT_ASSERT(!iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 1);
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iter, kEvent, 2, 3, 3, 0, 1));  // event 2:1:1
          CPPUNIT_ASSERT(iter.entry() == 1);
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iter, kRun, 4, 5, 5, 0, 1));  // run 1
          CPPUNIT_ASSERT(!iter.shouldProcessRun());
          CPPUNIT_ASSERT(iter.entry() == 0);
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iter, kLumi, 4, 5, 5, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(!iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 0);
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iter, kLumi, 4, 6, 5, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(!iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 0);
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iter, kEvent, 4, 6, 5, 0, 1));  // event 1:1:2
          CPPUNIT_ASSERT(iter.entry() == 2);
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iter, kEvent, 4, 6, 6, 0, 1));  // event 1:1:3
          CPPUNIT_ASSERT(iter.entry() == 3);
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iter, kRun, 7, 10, 9, 0, 1));  // run 2
          CPPUNIT_ASSERT(iter.shouldProcessRun());
          CPPUNIT_ASSERT(iter.entry() == 1);
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iter, kRun, 8, 10, 9, 0, 1));  // run 2
          CPPUNIT_ASSERT(iter.shouldProcessRun());
          CPPUNIT_ASSERT(iter.entry() == 3);
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iter, kLumi, 8, 10, 9, 0, 1));  // lumi 2:1
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 1);
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iter, kLumi, 8, 11, 9, 0, 1));  // lumi 2:1
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 3);
          break;
        }
        case 15: {
          CPPUNIT_ASSERT(check(iter, kEvent, 8, 11, 9, 0, 1));  // event 2:1:2
          CPPUNIT_ASSERT(iter.entry() == 4);
          break;
        }
        case 16: {
          CPPUNIT_ASSERT(check(iter, kRun, 12, 14, 15, 0, 1));  // run 1
          CPPUNIT_ASSERT(iter.shouldProcessRun());
          CPPUNIT_ASSERT(iter.entry() == 0);
          break;
        }
        case 17: {
          CPPUNIT_ASSERT(check(iter, kRun, 13, 14, 15, 0, 1));  // run 1
          CPPUNIT_ASSERT(iter.shouldProcessRun());
          CPPUNIT_ASSERT(iter.entry() == 2);
          break;
        }
        case 18: {
          CPPUNIT_ASSERT(check(iter, kLumi, 13, 14, 15, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 0);
          break;
        }
        case 19: {
          CPPUNIT_ASSERT(check(iter, kLumi, 13, 15, 15, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 2);
          break;
        }
        case 20: {
          CPPUNIT_ASSERT(check(iter, kEvent, 13, 15, 15, 0, 1));  // event 1:1:4
          CPPUNIT_ASSERT(iter.entry() == 5);
          break;
        }
        default:
          CPPUNIT_ASSERT(false);
      }
    }
    CPPUNIT_ASSERT(i == 21);
  }
}

void TestIndexIntoFile3::testNonValidLumiInsideValidLumis() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.addEntry(fakePHID1, 1, 1, 1, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 0);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 1, 2, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 1);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 2);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 1, 3, 2);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 3);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 1, 4, 3);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 4);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 1, 1, 4);  // Event
  indexIntoFile.addEntry(fakePHID1, 2, 1, 0, 5);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 0);  // Run
  indexIntoFile.addEntry(fakePHID1, 1, 1, 5, 5);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 6);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 1);  // Run

  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  {
    edm::IndexIntoFile::IndexIntoFileItr iter = indexIntoFile.begin(IndexIntoFile::entryOrder);
    edm::IndexIntoFile::IndexIntoFileItr iterEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
    int i = 0;
    for (i = 0; iter != iterEnd; ++iter, ++i) {
      switch (i) {
        case 0: {
          CPPUNIT_ASSERT(check(iter, kRun, 0, 2, 1, 0, 1));  // run 1
          CPPUNIT_ASSERT(iter.size() == 11);
          CPPUNIT_ASSERT(iter.indexedSize() == 14);
          CPPUNIT_ASSERT(!iter.shouldProcessRun());
          break;
        }
        case 1: {
          CPPUNIT_ASSERT(check(iter, kLumi, 0, 2, 1, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(!iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.lumiIterationStartingIndex(2));
          break;
        }
        case 2: {
          CPPUNIT_ASSERT(check(iter, kLumi, 0, 3, 1, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(!iter.shouldProcessLumi());
          CPPUNIT_ASSERT(!iter.lumiIterationStartingIndex(3));
          break;
        }
        case 3: {
          CPPUNIT_ASSERT(check(iter, kLumi, 0, 4, 1, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(!iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.lumiIterationStartingIndex(4));
          break;
        }
        case 4: {
          CPPUNIT_ASSERT(check(iter, kEvent, 0, 4, 1, 0, 1));  // event 1:1:1
          CPPUNIT_ASSERT(iter.entry() == 0);
          break;
        }
        case 5: {
          CPPUNIT_ASSERT(check(iter, kEvent, 0, 4, 2, 0, 1));  // event 1:1:2
          CPPUNIT_ASSERT(iter.entry() == 1);
          break;
        }
        case 6: {
          CPPUNIT_ASSERT(check(iter, kEvent, 0, 4, 3, 0, 1));  // event 1:1:3
          CPPUNIT_ASSERT(iter.entry() == 2);
          break;
        }
        case 7: {
          CPPUNIT_ASSERT(check(iter, kEvent, 0, 4, 4, 0, 1));  // event 1:1:4
          CPPUNIT_ASSERT(iter.entry() == 3);
          break;
        }
        case 8: {
          CPPUNIT_ASSERT(check(iter, kRun, 5, 6, 6, 0, 1));  // run 2
          CPPUNIT_ASSERT(iter.shouldProcessRun());
          break;
        }
        case 9: {
          CPPUNIT_ASSERT(check(iter, kLumi, 5, 6, 6, 0, 1));  // lumi 2:1
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          break;
        }
        case 10: {
          CPPUNIT_ASSERT(check(iter, kEvent, 5, 6, 6, 0, 1));  // event 2:1:1
          CPPUNIT_ASSERT(iter.entry() == 4);
          break;
        }
        case 11: {
          CPPUNIT_ASSERT(check(iter, kRun, 7, 8, -1, 0, 0));  // run 1
          CPPUNIT_ASSERT(iter.shouldProcessRun());
          break;
        }
        case 12: {
          CPPUNIT_ASSERT(check(iter, kLumi, 7, 8, -1, 0, 0));  // lumi 1:2
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          break;
        }
        case 13: {
          CPPUNIT_ASSERT(check(iter, kLumi, 7, 9, -1, 0, 0));  // lumi 1:2
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          break;
        }
        case 14: {
          CPPUNIT_ASSERT(check(iter, kLumi, 7, 10, -1, 0, 0));  // lumi 1:2
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          break;
        }
        case 15: {
          CPPUNIT_ASSERT(check(iter, kLumi, 7, 11, 13, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 1);
          break;
        }
        case 16: {
          CPPUNIT_ASSERT(check(iter, kLumi, 7, 12, 13, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 4);
          break;
        }
        case 17: {
          CPPUNIT_ASSERT(check(iter, kLumi, 7, 13, 13, 0, 1));  // lumi 1:1
          CPPUNIT_ASSERT(iter.shouldProcessLumi());
          CPPUNIT_ASSERT(iter.entry() == 6);
          break;
        }
        case 18: {
          CPPUNIT_ASSERT(check(iter, kEvent, 7, 13, 13, 0, 1));  // event 1:1:5
          CPPUNIT_ASSERT(iter.entry() == 5);
          break;
        }
      }
    }
    CPPUNIT_ASSERT(i == 19);

    skipEventBackward(iter);
    checkSkipped(0, 1, 1, 5);
    CPPUNIT_ASSERT(check(iter, kRun, 7, 11, 13, 0, 1));

    skipEventBackward(iter);
    checkSkipped(0, 2, 1, 4);
    CPPUNIT_ASSERT(check(iter, kRun, 5, 6, 6, 0, 1));

    skipEventBackward(iter);
    checkSkipped(0, 1, 1, 3);
    CPPUNIT_ASSERT(check(iter, kRun, 0, 2, 4, 0, 1));

    skipEventBackward(iter);
    checkSkipped(0, 1, 1, 2);
    CPPUNIT_ASSERT(check(iter, kRun, 0, 2, 3, 0, 1));

    skipEventBackward(iter);
    checkSkipped(0, 1, 1, 1);
    CPPUNIT_ASSERT(check(iter, kRun, 0, 2, 2, 0, 1));

    skipEventBackward(iter);
    checkSkipped(0, 1, 1, 0);
    CPPUNIT_ASSERT(check(iter, kRun, 0, 2, 1, 0, 1));

    skipEventBackward(iter);
    checkSkipped(-1, 0, 0, -1);
    CPPUNIT_ASSERT(check(iter, kRun, 0, 2, 1, 0, 1));

    iter.advanceToNextRun();
    CPPUNIT_ASSERT(check(iter, kRun, 5, 6, 6, 0, 1));  // run 2
    iter.advanceToNextRun();
    CPPUNIT_ASSERT(check(iter, kRun, 7, 8, -1, 0, 0));  // run 1
    iter.advanceToNextRun();
    CPPUNIT_ASSERT(check(iter, kEnd, -1, -1, -1, 0, 0));

    iter = indexIntoFile.begin(IndexIntoFile::entryOrder);
    iter.advanceToNextLumiOrRun();
    CPPUNIT_ASSERT(check(iter, kLumi, 0, 2, 1, 0, 1));  // lumi 1:1
    iter.advanceToNextLumiOrRun();
    CPPUNIT_ASSERT(check(iter, kRun, 5, 6, 6, 0, 1));  // run 2
    iter.advanceToNextLumiOrRun();
    CPPUNIT_ASSERT(check(iter, kLumi, 5, 6, 6, 0, 1));  // lumi 2:1
    iter.advanceToNextLumiOrRun();
    CPPUNIT_ASSERT(check(iter, kRun, 7, 8, -1, 0, 0));  // run 1
    iter.advanceToNextLumiOrRun();
    CPPUNIT_ASSERT(check(iter, kLumi, 7, 8, -1, 0, 0));  // lumi 1:2
    iter.advanceToNextLumiOrRun();
    CPPUNIT_ASSERT(check(iter, kLumi, 7, 11, 13, 0, 1));  // lumi 1:1
    iter.advanceToNextLumiOrRun();
    CPPUNIT_ASSERT(check(iter, kEnd, -1, -1, -1, 0, 0));
  }
}
