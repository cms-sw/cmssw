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

class TestIndexIntoFile2: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestIndexIntoFile2);  
  CPPUNIT_TEST(testAddEntryAndFixAndSort);
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

  void testAddEntryAndFixAndSort();

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
CPPUNIT_TEST_SUITE_REGISTRATION(TestIndexIntoFile2);

void TestIndexIntoFile2::testAddEntryAndFixAndSort() {
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

  eventEntries.emplace_back(10, 2);
  eventEntries.emplace_back(9, 3);
  eventEntries.emplace_back(8, 6);
  eventEntries.emplace_back(7, 0);
  eventEntries.emplace_back(6, 1);
  eventEntries.emplace_back(5, 4);
  eventEntries.emplace_back(4, 5);
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
