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

class TestIndexIntoFile2 {
public:
  static const IndexIntoFile::EntryType kRun = IndexIntoFile::kRun;
  static const IndexIntoFile::EntryType kLumi = IndexIntoFile::kLumi;
  static const IndexIntoFile::EntryType kEvent = IndexIntoFile::kEvent;
  static const IndexIntoFile::EntryType kEnd = IndexIntoFile::kEnd;

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

  void testAddEntryAndFixAndSort();
  void testAddEntryAndFixAndSort2();

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

  // This is a helper class for IndexIntoFile.
  class TestEventFinder : public IndexIntoFile::EventFinder {
  public:
    explicit TestEventFinder() {}
    virtual ~TestEventFinder() {}
    virtual EventNumber_t getEventNumberOfEntry(IndexIntoFile::EntryNumber_t entry) const {
      return testData_.at(entry);
    }
    void push_back(EventNumber_t e) { testData_.push_back(e); }

  private:
    std::vector<EventNumber_t> testData_;
  };
};

///registration of the test so that the runner can find it

bool TestIndexIntoFile2::check(edm::IndexIntoFile::IndexIntoFileItr const& iter,
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

void TestIndexIntoFile2::testAddEntryAndFixAndSort() {
  edm::IndexIntoFile indexIntoFile;
  indexIntoFile.fillEventNumbersOrEntries(true, true);  // Should do nothing, it is empty at this point

  indexIntoFile.addEntry(fakePHID1, 11, 12, 7, 0);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 6, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 0, 0);  // Lumi
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].orderPHIDRunLumi() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].beginEvents() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].endEvents() == 2);

  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 0);  // Run
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].orderPHIDRun() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID2, 11, 12, 10, 2);  // Event
  indexIntoFile.addEntry(fakePHID2, 11, 12, 9, 3);   // Event
  indexIntoFile.addEntry(fakePHID2, 11, 12, 0, 1);   // Lumi
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 3);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].processHistoryIDIndex() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[1] == fakePHID2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.setRunOrLumiEntries()[2].orderPHIDRunLumi() == 1);
  REQUIRE(indexIntoFile.setRunOrLumiEntries()[2].beginEvents() == 2);
  REQUIRE(indexIntoFile.setRunOrLumiEntries()[2].endEvents() == 4);

  indexIntoFile.addEntry(fakePHID2, 11, 0, 0, 1);  // Run
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 4);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].processHistoryIDIndex() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[1] == fakePHID2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].orderPHIDRun() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID1, 11, 12, 5, 4);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 4, 5);  // Event
  indexIntoFile.addEntry(fakePHID1, 11, 12, 0, 2);  // Lumi
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 5);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].orderPHIDRunLumi() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].beginEvents() == 4);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].endEvents() == 6);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].entry() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].run() == 11);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].lumi() == 12);

  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 2);  // Run
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 6);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  IndexIntoFile::RunOrLumiEntry const& runOrLumiEntry = indexIntoFile.runOrLumiEntries()[5];
  REQUIRE(runOrLumiEntry.processHistoryIDIndex() == 0);
  REQUIRE(runOrLumiEntry.orderPHIDRun() == 0);
  REQUIRE(runOrLumiEntry.orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  REQUIRE(runOrLumiEntry.beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(runOrLumiEntry.endEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(runOrLumiEntry.entry() == 2);
  REQUIRE(runOrLumiEntry.run() == 11);
  REQUIRE(runOrLumiEntry.lumi() == IndexIntoFile::invalidLumi);

  indexIntoFile.addEntry(fakePHID1, 1, 3, 0, 3);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 2, 8, 6);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 4);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 5);  // Lumi
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 3);  // Run

  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  REQUIRE(indexIntoFile.runOrLumiEntries()[0].orderPHIDRun() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].orderPHIDRun() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].orderPHIDRun() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].orderPHIDRun() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].orderPHIDRun() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].orderPHIDRun() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].orderPHIDRun() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].orderPHIDRun() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].orderPHIDRun() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].orderPHIDRun() == 2);

  REQUIRE(indexIntoFile.runOrLumiEntries()[0].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].orderPHIDRunLumi() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].orderPHIDRunLumi() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].orderPHIDRunLumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].orderPHIDRunLumi() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].orderPHIDRunLumi() == 3);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].orderPHIDRunLumi() == 3);

  REQUIRE(indexIntoFile.runOrLumiEntries()[0].entry() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].entry() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].entry() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].entry() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].entry() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].entry() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].entry() == 3);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].entry() == 3);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].entry() == 4);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].entry() == 5);

  REQUIRE(indexIntoFile.processHistoryID(0) == fakePHID1);
  REQUIRE(indexIntoFile.processHistoryID(1) == fakePHID2);

  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.processHistoryIDs()[1] == fakePHID2);

  std::vector<ProcessHistoryID> sourcePHIDs;
  sourcePHIDs.push_back(fakePHID3);
  sourcePHIDs.push_back(fakePHID2);

  indexIntoFile.fixIndexes(sourcePHIDs);

  REQUIRE(sourcePHIDs.size() == 3);
  REQUIRE(sourcePHIDs[0] == fakePHID3);
  REQUIRE(sourcePHIDs[1] == fakePHID2);
  REQUIRE(sourcePHIDs[2] == fakePHID1);
  REQUIRE(sourcePHIDs == indexIntoFile.processHistoryIDs());

  REQUIRE(indexIntoFile.runOrLumiEntries()[0].processHistoryIDIndex() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].processHistoryIDIndex() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].processHistoryIDIndex() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].processHistoryIDIndex() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].processHistoryIDIndex() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].processHistoryIDIndex() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].processHistoryIDIndex() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].processHistoryIDIndex() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].processHistoryIDIndex() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].processHistoryIDIndex() == 2);

  std::vector<ProcessHistoryID>& phids = indexIntoFile.setProcessHistoryIDs();
  phids.push_back(nullPHID);
  REQUIRE(nullPHID == indexIntoFile.processHistoryID(3));

  unsigned count = 0;
  IndexIntoFile::SortedRunOrLumiItr runOrLumi = indexIntoFile.beginRunOrLumi();
  for (IndexIntoFile::SortedRunOrLumiItr endRunOrLumi = indexIntoFile.endRunOrLumi(); runOrLumi != endRunOrLumi;
       ++runOrLumi, ++count) {
    IndexIntoFile::RunOrLumiIndexes const& indexes = runOrLumi.runOrLumiIndexes();
    switch (count) {
      case 0:
        REQUIRE(indexes.indexToGetEntry() == 4);
        REQUIRE(indexes.beginEventNumbers() == -1);
        REQUIRE(indexes.endEventNumbers() == -1);
        break;
      case 1:
        REQUIRE(indexes.indexToGetEntry() == 5);
        REQUIRE(indexes.beginEventNumbers() == 0);
        REQUIRE(indexes.endEventNumbers() == 2);
        break;
      case 2:
        REQUIRE(indexes.indexToGetEntry() == 6);
        REQUIRE(indexes.beginEventNumbers() == -1);
        REQUIRE(indexes.endEventNumbers() == -1);
        break;
      case 3:
        REQUIRE(indexes.indexToGetEntry() == 8);
        REQUIRE(indexes.beginEventNumbers() == 2);
        REQUIRE(indexes.endEventNumbers() == 3);
        break;
      case 4:
        REQUIRE(indexes.indexToGetEntry() == 9);
        REQUIRE(indexes.beginEventNumbers() == 2);
        REQUIRE(indexes.endEventNumbers() == 3);
        break;
      case 5:
        REQUIRE(indexes.indexToGetEntry() == 7);
        REQUIRE(indexes.beginEventNumbers() == 3);
        REQUIRE(indexes.endEventNumbers() == 3);
        break;
      case 6:
        REQUIRE(indexes.indexToGetEntry() == 0);
        REQUIRE(indexes.beginEventNumbers() == -1);
        REQUIRE(indexes.endEventNumbers() == -1);
        break;
      case 7:
        REQUIRE(indexes.indexToGetEntry() == 1);
        REQUIRE(indexes.beginEventNumbers() == -1);
        REQUIRE(indexes.endEventNumbers() == -1);
        break;
      case 8:
        REQUIRE(indexes.indexToGetEntry() == 2);
        REQUIRE(indexes.beginEventNumbers() == 3);
        REQUIRE(indexes.endEventNumbers() == 7);
        break;
      case 9:
        REQUIRE(indexes.indexToGetEntry() == 3);
        REQUIRE(indexes.beginEventNumbers() == 3);
        REQUIRE(indexes.endEventNumbers() == 7);
        break;
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

  REQUIRE(eventNumbers[0] == 9);
  REQUIRE(eventNumbers[1] == 10);
  REQUIRE(eventNumbers[2] == 8);
  REQUIRE(eventNumbers[3] == 4);
  REQUIRE(eventNumbers[4] == 5);
  REQUIRE(eventNumbers[5] == 6);
  REQUIRE(eventNumbers[6] == 7);

  std::vector<IndexIntoFile::EventEntry>& eventEntries = indexIntoFile.eventEntries();

  eventEntries.emplace_back(10, 2);
  eventEntries.emplace_back(9, 3);
  eventEntries.emplace_back(8, 6);
  eventEntries.emplace_back(7, 0);
  eventEntries.emplace_back(6, 1);
  eventEntries.emplace_back(5, 4);
  eventEntries.emplace_back(4, 5);
  indexIntoFile.sortEventEntries();

  REQUIRE(eventEntries[0].event() == 9);
  REQUIRE(eventEntries[1].event() == 10);
  REQUIRE(eventEntries[2].event() == 8);
  REQUIRE(eventEntries[3].event() == 4);
  REQUIRE(eventEntries[4].event() == 5);
  REQUIRE(eventEntries[5].event() == 6);
  REQUIRE(eventEntries[6].event() == 7);

  std::vector<EventNumber_t>().swap(eventNumbers);
  std::vector<IndexIntoFile::EventEntry>().swap(eventEntries);

  REQUIRE(indexIntoFile.numberOfEvents() == 7);

  TestEventFinder* ptr(new TestEventFinder);
  ptr->push_back(7);
  ptr->push_back(6);
  ptr->push_back(10);
  ptr->push_back(9);
  ptr->push_back(5);
  ptr->push_back(4);
  ptr->push_back(8);

  std::shared_ptr<IndexIntoFile::EventFinder> shptr(ptr);
  indexIntoFile.setEventFinder(shptr);

  indexIntoFile.fillEventNumbers();
  REQUIRE(eventNumbers[0] == 9);
  REQUIRE(eventNumbers[1] == 10);
  REQUIRE(eventNumbers[2] == 8);
  REQUIRE(eventNumbers[3] == 4);
  REQUIRE(eventNumbers[4] == 5);
  REQUIRE(eventNumbers[5] == 6);
  REQUIRE(eventNumbers[6] == 7);

  indexIntoFile.fillEventEntries();
  REQUIRE(eventEntries[0].event() == 9);
  REQUIRE(eventEntries[1].event() == 10);
  REQUIRE(eventEntries[2].event() == 8);
  REQUIRE(eventEntries[3].event() == 4);
  REQUIRE(eventEntries[4].event() == 5);
  REQUIRE(eventEntries[5].event() == 6);
  REQUIRE(eventEntries[6].event() == 7);

  std::vector<EventNumber_t>().swap(eventNumbers);
  std::vector<IndexIntoFile::EventEntry>().swap(eventEntries);

  indexIntoFile.fillEventEntries();
  REQUIRE(eventEntries[0].event() == 9);
  REQUIRE(eventEntries[1].event() == 10);
  REQUIRE(eventEntries[2].event() == 8);
  REQUIRE(eventEntries[3].event() == 4);
  REQUIRE(eventEntries[4].event() == 5);
  REQUIRE(eventEntries[5].event() == 6);
  REQUIRE(eventEntries[6].event() == 7);

  indexIntoFile.fillEventNumbers();
  REQUIRE(eventNumbers[0] == 9);
  REQUIRE(eventNumbers[1] == 10);
  REQUIRE(eventNumbers[2] == 8);
  REQUIRE(eventNumbers[3] == 4);
  REQUIRE(eventNumbers[4] == 5);
  REQUIRE(eventNumbers[5] == 6);
  REQUIRE(eventNumbers[6] == 7);

  std::vector<EventNumber_t>().swap(eventNumbers);
  std::vector<IndexIntoFile::EventEntry>().swap(eventEntries);

  indexIntoFile.fillEventNumbersOrEntries(true, true);
  indexIntoFile.fillEventNumbersOrEntries(true, true);

  REQUIRE(eventNumbers[0] == 9);
  REQUIRE(eventNumbers[1] == 10);
  REQUIRE(eventNumbers[2] == 8);
  REQUIRE(eventNumbers[3] == 4);
  REQUIRE(eventNumbers[4] == 5);
  REQUIRE(eventNumbers[5] == 6);
  REQUIRE(eventNumbers[6] == 7);

  REQUIRE(eventEntries[0].event() == 9);
  REQUIRE(eventEntries[1].event() == 10);
  REQUIRE(eventEntries[2].event() == 8);
  REQUIRE(eventEntries[3].event() == 4);
  REQUIRE(eventEntries[4].event() == 5);
  REQUIRE(eventEntries[5].event() == 6);
  REQUIRE(eventEntries[6].event() == 7);

  std::vector<EventNumber_t>().swap(eventNumbers);
  std::vector<IndexIntoFile::EventEntry>().swap(eventEntries);

  std::vector<EventNumber_t>& unsortedEventNumbers = indexIntoFile.unsortedEventNumbers();
  REQUIRE(!unsortedEventNumbers.empty());
  indexIntoFile.doneFileInitialization();
  REQUIRE(unsortedEventNumbers.empty());
  REQUIRE(unsortedEventNumbers.capacity() == 0);
  unsortedEventNumbers.push_back(7);
  unsortedEventNumbers.push_back(6);
  unsortedEventNumbers.push_back(10);
  unsortedEventNumbers.push_back(9);
  unsortedEventNumbers.push_back(5);
  unsortedEventNumbers.push_back(4);
  unsortedEventNumbers.push_back(8);

  indexIntoFile.fillEventNumbersOrEntries(true, true);

  REQUIRE(eventNumbers[0] == 9);
  REQUIRE(eventNumbers[1] == 10);
  REQUIRE(eventNumbers[2] == 8);
  REQUIRE(eventNumbers[3] == 4);
  REQUIRE(eventNumbers[4] == 5);
  REQUIRE(eventNumbers[5] == 6);
  REQUIRE(eventNumbers[6] == 7);

  REQUIRE(eventEntries[0].event() == 9);
  REQUIRE(eventEntries[1].event() == 10);
  REQUIRE(eventEntries[2].event() == 8);
  REQUIRE(eventEntries[3].event() == 4);
  REQUIRE(eventEntries[4].event() == 5);
  REQUIRE(eventEntries[5].event() == 6);
  REQUIRE(eventEntries[6].event() == 7);

  indexIntoFile.inputFileClosed();
  REQUIRE(unsortedEventNumbers.empty());
  REQUIRE(unsortedEventNumbers.capacity() == 0);
  REQUIRE(eventEntries.capacity() == 0);
  REQUIRE(eventEntries.empty());
  REQUIRE(indexIntoFile.runOrLumiIndexes().capacity() == 0);
  REQUIRE(indexIntoFile.runOrLumiIndexes().empty());
  REQUIRE(indexIntoFile.transient_.eventFinder_.get() == 0);
}
void TestIndexIntoFile2::testAddEntryAndFixAndSort2() {
  // Similar to the last test, but this one focuses on the ordering
  // issues that can occur on boundaries between runs and boundaries
  // between lumis with concurrent lumis and concurrent runs.
  edm::IndexIntoFile indexIntoFile;

  indexIntoFile.addEntry(fakePHID1, 11, 1, 0, 0);  // Lumi
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].orderPHIDRunLumi() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].entry() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].run() == 11);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].lumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[0].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID1, 1, 1, 1, 0);  // Event
  indexIntoFile.addEntry(fakePHID2, 1, 1, 0, 1);  // Lumi
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 3);

  REQUIRE(indexIntoFile.runOrLumiEntries()[1].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].orderPHIDRunLumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].entry() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].run() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].lumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].beginEvents() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[1].endEvents() == 1);

  REQUIRE(indexIntoFile.runOrLumiEntries()[2].processHistoryIDIndex() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[1] == fakePHID2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].orderPHIDRunLumi() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].entry() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].run() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].lumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[2].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID1, 1, 1, 2, 1);  // Event
  indexIntoFile.addEntry(fakePHID1, 2, 1, 0, 2);  // Lumi

  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 5);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].orderPHIDRunLumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].entry() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].run() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].lumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].beginEvents() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[3].endEvents() == 2);

  REQUIRE(indexIntoFile.runOrLumiEntries()[4].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].orderPHIDRunLumi() == 3);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].entry() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].run() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].lumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[4].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID1, 1, 1, 3, 2);  // Event
  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 3);  // Lumi

  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 7);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].orderPHIDRunLumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].entry() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].run() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].lumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].beginEvents() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[5].endEvents() == 3);

  REQUIRE(indexIntoFile.runOrLumiEntries()[6].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].orderPHIDRunLumi() == 4);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].entry() == 3);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].run() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].lumi() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[6].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID1, 1, 2, 0, 4);  // Lumi

  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 8);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].orderPHIDRunLumi() == 4);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].entry() == 4);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].run() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].lumi() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[7].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID1, 1, 1, 4, 3);  // Event
  indexIntoFile.addEntry(fakePHID1, 3, 0, 0, 0);  // Run
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 9);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].orderPHIDRun() == 4);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].entry() == 0);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].run() == 3);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].lumi() == IndexIntoFile::invalidLumi);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[8].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID2, 1, 1, 1, 4);  // Event
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 10);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].processHistoryIDIndex() == 0);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[0] == fakePHID1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].orderPHIDRunLumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].entry() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].run() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].lumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].beginEvents() == 3);
  REQUIRE(indexIntoFile.runOrLumiEntries()[9].endEvents() == 4);

  indexIntoFile.addEntry(fakePHID2, 2, 1, 5, 5);  // Event
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 11);
  REQUIRE(indexIntoFile.runOrLumiEntries()[10].processHistoryIDIndex() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[1] == fakePHID2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[10].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[10].orderPHIDRunLumi() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[10].entry() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[10].run() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[10].lumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[10].beginEvents() == 4);
  REQUIRE(indexIntoFile.runOrLumiEntries()[10].endEvents() == 5);

  indexIntoFile.addEntry(fakePHID2, 2, 2, 1, 6);  // Event
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 12);
  REQUIRE(indexIntoFile.runOrLumiEntries()[11].processHistoryIDIndex() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[1] == fakePHID2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[11].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[11].orderPHIDRunLumi() == 5);
  REQUIRE(indexIntoFile.runOrLumiEntries()[11].entry() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[11].run() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[11].lumi() == 1);
  REQUIRE(indexIntoFile.runOrLumiEntries()[11].beginEvents() == 5);
  REQUIRE(indexIntoFile.runOrLumiEntries()[11].endEvents() == 6);

  indexIntoFile.addEntry(fakePHID2, 2, 2, 2, 7);  // Event
  indexIntoFile.addEntry(fakePHID2, 2, 2, 0, 5);  // Lumi
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 13);
  REQUIRE(indexIntoFile.runOrLumiEntries()[12].processHistoryIDIndex() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[1] == fakePHID2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[12].orderPHIDRun() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[12].orderPHIDRunLumi() == 6);
  REQUIRE(indexIntoFile.runOrLumiEntries()[12].entry() == 5);
  REQUIRE(indexIntoFile.runOrLumiEntries()[12].run() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[12].lumi() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[12].beginEvents() == 6);
  REQUIRE(indexIntoFile.runOrLumiEntries()[12].endEvents() == 8);

  indexIntoFile.addEntry(fakePHID1, 1, 1, 0, 6);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 1, 1, 0, 7);  // Lumi
  indexIntoFile.addEntry(fakePHID2, 2, 1, 0, 8);  // Lumi

  indexIntoFile.addEntry(fakePHID1, 11, 0, 0, 1);  // Run
  indexIntoFile.addEntry(fakePHID1, 1, 0, 0, 2);   // Run
  REQUIRE(indexIntoFile.runOrLumiEntries()[17].orderPHIDRun() == 1);
  indexIntoFile.addEntry(fakePHID2, 1, 0, 0, 3);  // Run

  indexIntoFile.addEntry(fakePHID2, 2, 0, 0, 4);  // Run
  REQUIRE(indexIntoFile.runOrLumiEntries().size() == 20);
  REQUIRE(indexIntoFile.runOrLumiEntries()[19].processHistoryIDIndex() == 1);
  REQUIRE(indexIntoFile.processHistoryIDs().size() == 2);
  REQUIRE(indexIntoFile.processHistoryIDs()[1] == fakePHID2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[19].orderPHIDRun() == 5);
  REQUIRE(indexIntoFile.runOrLumiEntries()[19].orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[19].entry() == 4);
  REQUIRE(indexIntoFile.runOrLumiEntries()[19].run() == 2);
  REQUIRE(indexIntoFile.runOrLumiEntries()[19].lumi() == IndexIntoFile::invalidLumi);
  REQUIRE(indexIntoFile.runOrLumiEntries()[19].beginEvents() == IndexIntoFile::invalidEntry);
  REQUIRE(indexIntoFile.runOrLumiEntries()[19].endEvents() == IndexIntoFile::invalidEntry);

  indexIntoFile.addEntry(fakePHID1, 2, 0, 0, 5);  // Run

  indexIntoFile.sortVector_Run_Or_Lumi_Entries();

  int index = 0;
  for (auto const& entry : indexIntoFile.runOrLumiEntries()) {
    switch (index) {
      case 0:
        REQUIRE(entry.orderPHIDRun() == 0);
        REQUIRE(entry.orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
        REQUIRE(entry.entry() == 1);
        break;
      case 1:
        REQUIRE(entry.orderPHIDRun() == 0);
        REQUIRE(entry.orderPHIDRunLumi() == 0);
        REQUIRE(entry.entry() == 0);
        break;
      case 2:
        REQUIRE(entry.orderPHIDRun() == 1);
        REQUIRE(entry.orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
        REQUIRE(entry.entry() == 2);
        break;
      case 3:
        REQUIRE(entry.orderPHIDRun() == 1);
        REQUIRE(entry.orderPHIDRunLumi() == 1);
        REQUIRE(entry.entry() == IndexIntoFile::invalidEntry);
        break;
      case 4:
        REQUIRE(entry.orderPHIDRun() == 1);
        REQUIRE(entry.orderPHIDRunLumi() == 1);
        REQUIRE(entry.entry() == IndexIntoFile::invalidEntry);
        break;
      case 5:
        REQUIRE(entry.orderPHIDRun() == 1);
        REQUIRE(entry.orderPHIDRunLumi() == 1);
        REQUIRE(entry.entry() == IndexIntoFile::invalidEntry);
        break;
      case 6:
        REQUIRE(entry.orderPHIDRun() == 1);
        REQUIRE(entry.orderPHIDRunLumi() == 1);
        REQUIRE(entry.entry() == IndexIntoFile::invalidEntry);
        break;
      case 7:
        REQUIRE(entry.orderPHIDRun() == 1);
        REQUIRE(entry.orderPHIDRunLumi() == 1);
        REQUIRE(entry.entry() == 6);
        break;
      case 8:
        REQUIRE(entry.orderPHIDRun() == 1);
        REQUIRE(entry.orderPHIDRunLumi() == 4);
        REQUIRE(entry.entry() == 3);
        break;
      case 9:
        REQUIRE(entry.orderPHIDRun() == 1);
        REQUIRE(entry.orderPHIDRunLumi() == 4);
        REQUIRE(entry.entry() == 4);
        break;
      case 10:
        REQUIRE(entry.orderPHIDRun() == 2);
        REQUIRE(entry.orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
        REQUIRE(entry.entry() == 3);
        break;
      case 11:
        REQUIRE(entry.orderPHIDRun() == 2);
        REQUIRE(entry.orderPHIDRunLumi() == 2);
        REQUIRE(entry.entry() == IndexIntoFile::invalidEntry);
        break;
      case 12:
        REQUIRE(entry.orderPHIDRun() == 2);
        REQUIRE(entry.orderPHIDRunLumi() == 2);
        REQUIRE(entry.entry() == 1);
        break;
      case 13:
        REQUIRE(entry.orderPHIDRun() == 2);
        REQUIRE(entry.orderPHIDRunLumi() == 2);
        REQUIRE(entry.entry() == 7);
        break;
      case 14:
        REQUIRE(entry.orderPHIDRun() == 3);
        REQUIRE(entry.orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
        REQUIRE(entry.entry() == 5);
        break;
      case 15:
        REQUIRE(entry.orderPHIDRun() == 3);
        REQUIRE(entry.orderPHIDRunLumi() == 3);
        REQUIRE(entry.entry() == 2);
        break;
      case 16:
        REQUIRE(entry.orderPHIDRun() == 4);
        REQUIRE(entry.orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
        REQUIRE(entry.entry() == 0);
        break;
      case 17:
        REQUIRE(entry.orderPHIDRun() == 5);
        REQUIRE(entry.orderPHIDRunLumi() == IndexIntoFile::invalidEntry);
        REQUIRE(entry.entry() == 4);
        break;
      case 18:
        REQUIRE(entry.orderPHIDRun() == 5);
        REQUIRE(entry.orderPHIDRunLumi() == 5);
        REQUIRE(entry.entry() == IndexIntoFile::invalidEntry);
        break;
      case 19:
        REQUIRE(entry.orderPHIDRun() == 5);
        REQUIRE(entry.orderPHIDRunLumi() == 5);
        REQUIRE(entry.entry() == 8);
        break;
      case 20:
        REQUIRE(entry.orderPHIDRun() == 5);
        REQUIRE(entry.orderPHIDRunLumi() == 6);
        REQUIRE(entry.entry() == 5);
        break;
    }
    ++index;
  }
  edm::IndexIntoFile::IndexIntoFileItr iterEntry = indexIntoFile.begin(IndexIntoFile::entryOrder);
  edm::IndexIntoFile::IndexIntoFileItr iterEntryEnd = indexIntoFile.end(IndexIntoFile::entryOrder);
  int i = 0;
  for (; iterEntry != iterEntryEnd; ++iterEntry, ++i) {
    if (i == 0)  // PHID1:Run3
      REQUIRE(check(iterEntry, kRun, 0, -1, -1, 0, 0));
    else if (i == 1)  // PHID1:Run11
      REQUIRE(check(iterEntry, kRun, 1, 2, -1, 0, 0));
    else if (i == 2)
      REQUIRE(check(iterEntry, kLumi, 1, 2, -1, 0, 0));
    else if (i == 3)  // PHID1:Run1
      REQUIRE(check(iterEntry, kRun, 3, 4, -1, 0, 0));
    else if (i == 4)
      REQUIRE(check(iterEntry, kLumi, 3, 4, -1, 0, 0));
    else if (i == 5)
      REQUIRE(check(iterEntry, kLumi, 3, 5, -1, 0, 0));
    else if (i == 6)
      REQUIRE(check(iterEntry, kLumi, 3, 10, 6, 0, 1));
    else if (i == 7)
      REQUIRE(check(iterEntry, kEvent, 3, 10, 6, 0, 1));
    else if (i == 8)
      REQUIRE(check(iterEntry, kEvent, 3, 10, 7, 0, 1));
    else if (i == 9)
      REQUIRE(check(iterEntry, kEvent, 3, 10, 8, 0, 1));
    else if (i == 10)
      REQUIRE(check(iterEntry, kEvent, 3, 10, 9, 0, 1));
    else if (i == 11)  // PHID2:Run1
      REQUIRE(check(iterEntry, kRun, 11, 13, 12, 0, 1));
    else if (i == 12)
      REQUIRE(check(iterEntry, kLumi, 11, 13, 12, 0, 1));
    else if (i == 13)
      REQUIRE(check(iterEntry, kLumi, 11, 14, 12, 0, 1));
    else if (i == 14)
      REQUIRE(check(iterEntry, kEvent, 11, 14, 12, 0, 1));
    else if (i == 15)  // PHID2:Run2
      REQUIRE(check(iterEntry, kRun, 15, 17, 16, 0, 1));
    else if (i == 16)
      REQUIRE(check(iterEntry, kLumi, 15, 17, 16, 0, 1));
    else if (i == 17)
      REQUIRE(check(iterEntry, kEvent, 15, 17, 16, 0, 1));
    else if (i == 18)
      REQUIRE(check(iterEntry, kLumi, 15, 18, 18, 0, 2));
    else if (i == 19)
      REQUIRE(check(iterEntry, kEvent, 15, 18, 18, 0, 2));
    else if (i == 20)
      REQUIRE(check(iterEntry, kEvent, 15, 18, 18, 1, 2));
    else if (i == 21)  // PHID1:Run2
      REQUIRE(check(iterEntry, kRun, 19, 20, -1, 0, 0));
    else if (i == 22)
      REQUIRE(check(iterEntry, kLumi, 19, 20, -1, 0, 0));
    else
      REQUIRE(false);
  }
  REQUIRE(i == 23);
};  // TestIndexIntoFile2

TEST_CASE("IndexIntoFile2", "[indexIntoFile2_t]") {
  TestIndexIntoFile2 fixture;
  fixture.setUp();

  SECTION("testAddEntryAndFixAndSort") { fixture.testAddEntryAndFixAndSort(); }
  SECTION("testAddEntryAndFixAndSort2") { fixture.testAddEntryAndFixAndSort2(); }
}
