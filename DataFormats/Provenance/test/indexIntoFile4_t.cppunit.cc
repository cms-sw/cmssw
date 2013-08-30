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

class TestIndexIntoFile4: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestIndexIntoFile4);  
  CPPUNIT_TEST(testFind);
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

  void testFind();
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

  void checkSkipped(int phIndexOfSkippedEvent,
                    RunNumber_t runOfSkippedEvent,
                    LuminosityBlockNumber_t lumiOfSkippedEvent,
                    IndexIntoFile::EntryNumber_t skippedEventEntry);

  void skipEventBackward(edm::IndexIntoFile::IndexIntoFileItr & iter);

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
CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexIntoFile4);

void TestIndexIntoFile4::check(edm::IndexIntoFile::IndexIntoFileItr const& iter,
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

void TestIndexIntoFile4::checkSkipped(int phIndexOfSkippedEvent,
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

void TestIndexIntoFile4::skipEventBackward(edm::IndexIntoFile::IndexIntoFileItr & iter) {
  iter.skipEventBackward(skipped_.phIndexOfSkippedEvent_,
                         skipped_.runOfSkippedEvent_,
                         skipped_.lumiOfSkippedEvent_,
                         skipped_.skippedEventEntry_);
}

void TestIndexIntoFile4::testFind() {

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
      eventEntries.emplace_back(7, 0);
      eventEntries.emplace_back(6, 1);
      eventEntries.emplace_back(5, 2);
      eventEntries.emplace_back(4, 3);
      eventEntries.emplace_back(7, 4);
      eventEntries.emplace_back(6, 5);
      eventEntries.emplace_back(5, 6);
      eventEntries.emplace_back(4, 7);
      eventEntries.emplace_back(7, 8);
      eventEntries.emplace_back(6, 9);
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

void TestIndexIntoFile4::testReduce() {
  // This test is implemented in FWCore/Integration/test/ProcessHistory_t.cpp
  // because of dependency issues.
}
