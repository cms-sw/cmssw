#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <ostream>
#include <iomanip>

namespace edm {

  int const IndexIntoFile::invalidIndex;
  RunNumber_t const IndexIntoFile::invalidRun;
  LuminosityBlockNumber_t const IndexIntoFile::invalidLumi;
  EventNumber_t const IndexIntoFile::invalidEvent;
  IndexIntoFile::EntryNumber_t const IndexIntoFile::invalidEntry;

  IndexIntoFile::IndexIntoFile() : transients_(),
                                   processHistoryIDs_(),
                                   runOrLumiEntries_() {
  }

  IndexIntoFile::~IndexIntoFile() {
  }

  ProcessHistoryID const& IndexIntoFile::processHistoryID(int i) const {
    return processHistoryIDs_.at(i);
  }

  std::vector<ProcessHistoryID> const& IndexIntoFile::processHistoryIDs() const {
    return processHistoryIDs_;
  }

  void
  IndexIntoFile::addEntry(ProcessHistoryID const& processHistoryID,
                          RunNumber_t run,
                          LuminosityBlockNumber_t lumi,
                          EventNumber_t event,
                          EntryNumber_t entry) {
    int index = 0;
    // First see if the ProcessHistoryID is the same as the previous one.
    // This is just a performance optimization.  We expect to usually get
    // many in a row that are the same.
    if (previousAddedIndex() != invalidIndex &&
        processHistoryID == processHistoryIDs_[previousAddedIndex()]) {
      index = previousAddedIndex();
    }
    // If it was not the same as the previous one then search through the
    // entire vector.  If it is not there, it needs to be added at the
    // end.
    else {
      index = 0;
      while (index < static_cast<int>(processHistoryIDs_.size()) && 
             processHistoryIDs_[index] != processHistoryID) {
        ++index;        
      }
      if (index == static_cast<int>(processHistoryIDs_.size())) {
        processHistoryIDs_.push_back(processHistoryID);
      }
    }
    previousAddedIndex() = index;

    assert((currentRun() == run && currentIndex() == index) || currentRun() == invalidRun);
    if (lumi == invalidLumi) {
      if (currentLumi() != invalidLumi) {
        throw Exception(errors::LogicError)
          << "In IndexIntoFile::addEntry. Entries were added in illegal order.\n"
          << "This means the IndexIntoFile product in the output file will be corrupted.\n"
          << "The output file will be unusable for most purposes.\n"
          << "If this occurs after an unrelated exception was thrown in\n"
          << "endLuminosityBlock or endRun then ignore this exception and fix\n"
          << "the primary exception. This is an expected side effect.\n"
          << "Otherwise please report this to the core framework developers\n";
      }
      currentIndex() = invalidIndex;
      currentRun() = invalidRun;
      currentLumi() = invalidLumi;
      std::pair<IndexRunKey, EntryNumber_t> firstRunEntry(IndexRunKey(index, run), entry);
      runToFirstEntry().insert(firstRunEntry);
      RunOrLumiEntry runEntry(runToFirstEntry()[IndexRunKey(index, run)], invalidEntry, entry, index, run, lumi, invalidEntry, invalidEntry);
      runOrLumiEntries_.push_back(runEntry);
    }
    else {
      assert(currentLumi() == lumi || currentLumi() == invalidLumi);
      if (currentRun() == invalidRun) {
        currentRun() = run;
        currentIndex() = index;
      }
      if (event == invalidEvent) {
        currentLumi() = invalidLumi;
        std::pair<IndexRunLumiKey, EntryNumber_t> firstLumiEntry(IndexRunLumiKey(index, run, lumi), entry);
        lumiToFirstEntry().insert(firstLumiEntry);
        RunOrLumiEntry lumiEntry(invalidEntry, lumiToFirstEntry()[IndexRunLumiKey(index, run, lumi)],
                                 entry, index, run, lumi, beginEvents(), endEvents());
        runOrLumiEntries_.push_back(lumiEntry);
        beginEvents() = invalidEntry;
        endEvents() = invalidEntry;
      }
      else {
	setNumberOfEvents(numberOfEvents() + 1);
        if (beginEvents() == invalidEntry) {
          currentLumi() = lumi;
          beginEvents() = entry;
          endEvents() = beginEvents() + 1;
        }
        else {
          assert(currentLumi() == lumi);
          assert(entry == endEvents());
          ++endEvents();
        }
      }
    }
  }

  void IndexIntoFile::fillRunOrLumiIndexes() const {
    if (runOrLumiEntries_.empty() || !runOrLumiIndexes().empty()) {
      return;
    }
    runOrLumiIndexes().reserve(runOrLumiEntries_.size());

    int index = 0;
    for (std::vector<RunOrLumiEntry>::const_iterator iter = runOrLumiEntries_.begin(),
	                                             iEnd = runOrLumiEntries_.end();
         iter != iEnd;
         ++iter, ++index) {
      runOrLumiIndexes().push_back(RunOrLumiIndexes(iter->processHistoryIDIndex(),
                                                    iter->run(),
                                                    iter->lumi(),
                                                    index));
    }
    stable_sort_all(runOrLumiIndexes());

    long long beginEventNumbers = 0;

    std::vector<RunOrLumiIndexes>::iterator beginOfLumi = runOrLumiIndexes().begin();
    std::vector<RunOrLumiIndexes>::iterator endOfLumi = beginOfLumi;
    std::vector<RunOrLumiIndexes>::iterator iEnd = runOrLumiIndexes().end();
    while (true) {
      while (beginOfLumi != iEnd && beginOfLumi->isRun()) {
        ++beginOfLumi;
      }
      if (beginOfLumi == iEnd) break;

      endOfLumi = beginOfLumi + 1;
      while (endOfLumi != iEnd &&
             beginOfLumi->processHistoryIDIndex() == endOfLumi->processHistoryIDIndex() &&
             beginOfLumi->run() == endOfLumi->run() &&
             beginOfLumi->lumi() == endOfLumi->lumi()) {
        ++endOfLumi;
      }
      int nEvents = 0;
      for (std::vector<RunOrLumiIndexes>::iterator iter = beginOfLumi;
           iter != endOfLumi;
	     ++iter) {
        if (runOrLumiEntries_[iter->indexToGetEntry()].beginEvents() != invalidEntry) {
          nEvents += runOrLumiEntries_[iter->indexToGetEntry()].endEvents() -
	             runOrLumiEntries_[iter->indexToGetEntry()].beginEvents();
        }
      }
      for (std::vector<RunOrLumiIndexes>::iterator iter = beginOfLumi;
           iter != endOfLumi;
	     ++iter) {
        iter->setBeginEventNumbers(beginEventNumbers);
        iter->setEndEventNumbers(beginEventNumbers + nEvents);
      }
      beginEventNumbers += nEvents;
      beginOfLumi = endOfLumi;
    }
    assert(runOrLumiIndexes().size() == runOrLumiEntries_.size());
  }

  void
  IndexIntoFile::fillEventNumbers() const {
    fillEventNumbersOrEntries(true, false);
  }

  void
  IndexIntoFile::fillEventEntries() const {
    fillEventNumbersOrEntries(false, true);
  }

  void
  IndexIntoFile::fillEventNumbersOrEntries(bool needEventNumbers, bool needEventEntries) const {
    if (numberOfEvents() == 0) {
      return;
    }

    if (needEventNumbers && !eventNumbers().empty()) {
      needEventNumbers = false;
    }

    if (needEventEntries && !eventEntries().empty()) {
      needEventEntries = false;
    }

    if (needEventNumbers && !eventEntries().empty()) {
      assert(numberOfEvents() == eventEntries().size());
      eventNumbers().reserve(eventEntries().size());
      for (std::vector<EventNumber_t>::size_type entry = 0U; entry < numberOfEvents(); ++entry) {
        eventNumbers().push_back(eventEntries()[entry].event());
      }
      return;
    }

    if (!needEventNumbers && !needEventEntries) {
      return;
    }

    fillUnsortedEventNumbers();

    if (needEventNumbers) {
      eventNumbers().resize(numberOfEvents(), IndexIntoFile::invalidEvent);
    }
    if (needEventEntries) {
      eventEntries().resize(numberOfEvents());
    }

    long long offset = 0;
    long long previousBeginEventNumbers = -1LL;

    for (SortedRunOrLumiItr runOrLumi = beginRunOrLumi(), runOrLumiEnd = endRunOrLumi();
         runOrLumi != runOrLumiEnd; ++runOrLumi) {

      if (runOrLumi.isRun()) continue;

      long long beginEventNumbers = 0;
      long long endEventNumbers = 0;
      EntryNumber_t beginEventEntry = -1LL;
      EntryNumber_t endEventEntry = -1LL;
      runOrLumi.getRange(beginEventNumbers, endEventNumbers, beginEventEntry, endEventEntry);

      // This is true each time one hits a new lumi section (except if the previous lumi had
      // no events, in which case the offset is still 0 anyway)
      if (beginEventNumbers != previousBeginEventNumbers) offset = 0;

      for (EntryNumber_t entry = beginEventEntry; entry != endEventEntry; ++entry) {
        if (needEventNumbers) {
          eventNumbers().at((entry - beginEventEntry) + offset + beginEventNumbers) = unsortedEventNumbers().at(entry);
        }
        if (needEventEntries) {
          eventEntries().at((entry - beginEventEntry) + offset + beginEventNumbers) =
            EventEntry(unsortedEventNumbers().at(entry), entry);
        }
      }

      previousBeginEventNumbers = beginEventNumbers;
      offset += endEventEntry - beginEventEntry;
    }
    if (needEventNumbers) {
      sortEvents();
      assert(numberOfEvents() == eventNumbers().size());
    }
    if (needEventEntries) {
      sortEventEntries();
      assert(numberOfEvents() == eventEntries().size());
    }
  }

  void 
  IndexIntoFile::fillUnsortedEventNumbers() const {
    if (numberOfEvents() == 0 || !unsortedEventNumbers().empty()) {
      return;
    }
    unsortedEventNumbers().reserve(numberOfEvents());

    // The main purpose for the existence of the unsortedEventNumbers
    // vector is that it can easily be filled by reading through
    // the EventAuxiliary branch in the same order as the TTree
    // entries. fillEventNumbersOrEntries can then use this information
    // instead of using getEventNumberOfEntry directly and reading
    // the branch in a different order.
    for (std::vector<EventNumber_t>::size_type entry = 0U; entry < numberOfEvents(); ++entry) {
      unsortedEventNumbers().push_back(getEventNumberOfEntry(entry));
    }
  }

  // We are closing the input file, but we need to keep event numbers.
  // We can delete the other transient collections by using the swap trick.

  void
  IndexIntoFile::inputFileClosed() const {
    std::vector<EventEntry>().swap(eventEntries());
    std::vector<RunOrLumiIndexes>().swap(runOrLumiIndexes());
    std::vector<EventNumber_t>().swap(unsortedEventNumbers());
    resetEventFinder();
  }

  void
  IndexIntoFile::doneFileInitialization() const {
    std::vector<EventNumber_t>().swap(unsortedEventNumbers());
  }

  void
  IndexIntoFile::fixIndexes(std::vector<ProcessHistoryID> & processHistoryIDs) {

    std::map<int, int> oldToNewIndex;
    for (std::vector<ProcessHistoryID>::const_iterator iter = processHistoryIDs_.begin(),
                                                       iEnd = processHistoryIDs_.end();
         iter != iEnd;
         ++iter) {
      std::vector<ProcessHistoryID>::const_iterator iterExisting =
        std::find(processHistoryIDs.begin(), processHistoryIDs.end(), *iter);
      if (iterExisting == processHistoryIDs.end()) {
        oldToNewIndex[iter - processHistoryIDs_.begin()] = processHistoryIDs.size();
        processHistoryIDs.push_back(*iter);
      }
      else {
        oldToNewIndex[iter - processHistoryIDs_.begin()] = iterExisting - processHistoryIDs.begin();
      }
    }
    processHistoryIDs_ = processHistoryIDs;

    for (std::vector<RunOrLumiEntry>::iterator iter = runOrLumiEntries_.begin(),
	                                       iEnd = runOrLumiEntries_.end();
         iter != iEnd;
         ++iter) {
      iter->setProcessHistoryIDIndex(oldToNewIndex[iter->processHistoryIDIndex()]);
    }
  }

  void IndexIntoFile::sortVector_Run_Or_Lumi_Entries() {
    for (std::vector<RunOrLumiEntry>::iterator iter = runOrLumiEntries_.begin(),
                                               iEnd = runOrLumiEntries_.end();
         iter != iEnd;
         ++iter) {
      std::map<IndexRunKey, EntryNumber_t>::const_iterator firstRunEntry = 
        runToFirstEntry().find(IndexRunKey(iter->processHistoryIDIndex(), iter->run()));
      if (firstRunEntry == runToFirstEntry().end()) {
        throw Exception(errors::LogicError)
          << "In IndexIntoFile::sortVector_Run_Or_Lumi_Entries. A run entry is missing.\n"
          << "This means the IndexIntoFile product in the output file will be corrupted.\n"
          << "The output file will be unusable for most purposes.\n"
          << "If this occurs after an unrelated exception was thrown in\n"
          << "endLuminosityBlock or endRun then ignore this exception and fix\n"
          << "the primary exception. This is an expected side effect.\n"
          << "Otherwise please report this to the core framework developers\n";
      }
      iter->setOrderPHIDRun(firstRunEntry->second);
    }
    stable_sort_all(runOrLumiEntries_);
  }

  void IndexIntoFile::sortEvents() const {
    fillRunOrLumiIndexes();
    std::vector<RunOrLumiIndexes>::iterator beginOfLumi = runOrLumiIndexes().begin();
    std::vector<RunOrLumiIndexes>::iterator endOfLumi = beginOfLumi;
    std::vector<RunOrLumiIndexes>::iterator iEnd = runOrLumiIndexes().end();
    while (true) {
      while (beginOfLumi != iEnd && beginOfLumi->isRun()) {
        ++beginOfLumi;
      }
      if (beginOfLumi == iEnd) break;

      endOfLumi = beginOfLumi + 1;
      while (endOfLumi != iEnd &&
             beginOfLumi->processHistoryIDIndex() == endOfLumi->processHistoryIDIndex() &&
             beginOfLumi->run() == endOfLumi->run() &&
             beginOfLumi->lumi() == endOfLumi->lumi()) {
        ++endOfLumi;
      }
      assert(beginOfLumi->endEventNumbers() >= 0);
      assert(beginOfLumi->endEventNumbers() <= static_cast<long long>(eventNumbers().size()));
      std::sort(eventNumbers().begin() + beginOfLumi->beginEventNumbers(),
                eventNumbers().begin() + beginOfLumi->endEventNumbers());
      beginOfLumi = endOfLumi;
    }
  }

  void IndexIntoFile::sortEventEntries() const {
    fillRunOrLumiIndexes();
    std::vector<RunOrLumiIndexes>::iterator beginOfLumi = runOrLumiIndexes().begin();
    std::vector<RunOrLumiIndexes>::iterator endOfLumi = beginOfLumi;
    std::vector<RunOrLumiIndexes>::iterator iEnd = runOrLumiIndexes().end();
    while (true) {
      while (beginOfLumi != iEnd && beginOfLumi->isRun()) {
        ++beginOfLumi;
      }
      if (beginOfLumi == iEnd) break;

      endOfLumi = beginOfLumi + 1;
      while (endOfLumi != iEnd &&
             beginOfLumi->processHistoryIDIndex() == endOfLumi->processHistoryIDIndex() &&
             beginOfLumi->run() == endOfLumi->run() &&
             beginOfLumi->lumi() == endOfLumi->lumi()) {
        ++endOfLumi;
      }
      assert(beginOfLumi->endEventNumbers() >= 0);
      assert(beginOfLumi->endEventNumbers() <=  static_cast<long long>(eventEntries().size()));
      std::sort(eventEntries().begin() + beginOfLumi->beginEventNumbers(),
                eventEntries().begin() + beginOfLumi->endEventNumbers());
      beginOfLumi = endOfLumi;
    }
  }

  IndexIntoFile::IndexIntoFileItr IndexIntoFile::begin(SortOrder sortOrder) const {
    if (empty()) {
      return end(sortOrder);
    }   
    IndexIntoFileItr iter(this,
                          sortOrder,
                          kRun,
                          0,
                          invalidIndex,
                          invalidIndex,
                          0,
                          0);
    iter.initializeRun();
    return iter;
  }

  IndexIntoFile::IndexIntoFileItr IndexIntoFile::end(SortOrder sortOrder) const {
    return IndexIntoFileItr(this,
                            sortOrder,
                            kEnd,
                            invalidIndex,
                            invalidIndex,
                            invalidIndex,
                            0,
                            0);
  }

  bool IndexIntoFile::iterationWillBeInEntryOrder(SortOrder sortOrder) const {
    EntryNumber_t maxEntry = invalidEntry;
    for(IndexIntoFileItr it = begin(sortOrder), itEnd = end(sortOrder); it != itEnd; ++it) {
      if(it.getEntryType() == kEvent) {
        if(it.entry() < maxEntry) {
	  return false;
        }
	maxEntry = it.entry();
      }
    }
    return true;
  }

  bool IndexIntoFile::empty() const {
    return runOrLumiEntries().empty();
  }

  IndexIntoFile::IndexIntoFileItr
  IndexIntoFile::findPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
    fillRunOrLumiIndexes();

    bool lumiMissing = (lumi == 0 && event != 0);

    std::vector<RunOrLumiIndexes>::const_iterator it;
    std::vector<RunOrLumiIndexes>::const_iterator iEnd = runOrLumiIndexes().end();
    std::vector<RunOrLumiIndexes>::const_iterator phEnd;

    // Loop over ranges of entries with the same ProcessHistoryID
    for (std::vector<RunOrLumiIndexes>::const_iterator phBegin = runOrLumiIndexes().begin();
         phBegin != iEnd;
         phBegin = phEnd) {

      RunOrLumiIndexes el(phBegin->processHistoryIDIndex(), run, lumi, 0);
      phEnd = std::upper_bound(phBegin, iEnd, el, Compare_Index());

      std::vector<RunOrLumiIndexes>::const_iterator iRun = std::lower_bound(phBegin, phEnd, el, Compare_Index_Run());

      if (iRun == phEnd || iRun->run() != run) continue;

      if (lumi == invalidLumi && event == invalidEvent) {
        IndexIntoFileItr indexItr(this,
                                  numericalOrder,
                                  kRun,
                                  iRun - runOrLumiIndexes().begin(),
                                  invalidIndex,
                                  invalidIndex,
                                  0,
                                  0);
        indexItr.initializeRun();
        return indexItr;
      }

      std::vector<RunOrLumiIndexes>::const_iterator iRunEnd = std::upper_bound(iRun, phEnd, el, Compare_Index_Run());
      if (!lumiMissing) {

        std::vector<RunOrLumiIndexes>::const_iterator iLumi = std::lower_bound(iRun, iRunEnd, el);
        if (iLumi == iRunEnd || iLumi->lumi() != lumi) continue;

        if (event == invalidEvent) {
          IndexIntoFileItr indexItr(this,
                                    numericalOrder,
                                    kRun,
                                    iRun - runOrLumiIndexes().begin(),
                                    iLumi - runOrLumiIndexes().begin(),
                                    invalidIndex,
                                    0,
                                    0);
          indexItr.initializeLumi();
          return indexItr;
        }

        long long beginEventNumbers = iLumi->beginEventNumbers();
        long long endEventNumbers = iLumi->endEventNumbers();
        if (beginEventNumbers >= endEventNumbers) continue;


        long long indexToEvent = 0;
        if (!eventEntries().empty()) {
          std::vector<EventEntry>::const_iterator eventIter = std::lower_bound(eventEntries().begin() + beginEventNumbers,
                                                                               eventEntries().begin() + endEventNumbers,
                                                                               EventEntry(event, invalidEntry));
          if (eventIter == (eventEntries().begin() + endEventNumbers) ||
              eventIter->event() != event) continue;

          indexToEvent = eventIter - eventEntries().begin() - beginEventNumbers;
        } else {
          fillEventNumbers();
          std::vector<EventNumber_t>::const_iterator eventIter = std::lower_bound(eventNumbers().begin() + beginEventNumbers,
                                                                                  eventNumbers().begin() + endEventNumbers,
                                                                                  event);
          if (eventIter == (eventNumbers().begin() + endEventNumbers) ||
              *eventIter != event) continue;

          indexToEvent = eventIter - eventNumbers().begin() - beginEventNumbers;
        }
        return IndexIntoFileItr(this,
                                numericalOrder,
                                kRun,
                                iRun - runOrLumiIndexes().begin(),
                                iLumi - runOrLumiIndexes().begin(),
                                iLumi - runOrLumiIndexes().begin(),
                                indexToEvent,
                                endEventNumbers - beginEventNumbers);
      }
      if (lumiMissing) {

        std::vector<RunOrLumiIndexes>::const_iterator iLumi = iRun;
        while (iLumi != iRunEnd && iLumi->lumi() == invalidLumi) {
          ++iLumi;
        }
        if (iLumi == iRunEnd) continue;

        std::vector<RunOrLumiIndexes>::const_iterator lumiEnd;
        for ( ;
             iLumi != iRunEnd;
             iLumi = lumiEnd) {

          RunOrLumiIndexes elWithLumi(phBegin->processHistoryIDIndex(), run, iLumi->lumi(), 0);
          lumiEnd = std::upper_bound(iLumi, iRunEnd, elWithLumi);

          long long beginEventNumbers = iLumi->beginEventNumbers();
          long long endEventNumbers = iLumi->endEventNumbers();
          if (beginEventNumbers >= endEventNumbers) continue;

          long long indexToEvent = 0;
          if (!eventEntries().empty()) {
            std::vector<EventEntry>::const_iterator eventIter = std::lower_bound(eventEntries().begin() + beginEventNumbers,
                                                                                 eventEntries().begin() + endEventNumbers,
                                                                                 EventEntry(event, invalidEntry));
            if (eventIter == (eventEntries().begin() + endEventNumbers) ||
                eventIter->event() != event) continue;
            indexToEvent = eventIter - eventEntries().begin() - beginEventNumbers;
          } else {
            fillEventNumbers();
            std::vector<EventNumber_t>::const_iterator eventIter = std::lower_bound(eventNumbers().begin() + beginEventNumbers,
                                                                                    eventNumbers().begin() + endEventNumbers,
                                                                                    event);
            if (eventIter == (eventNumbers().begin() + endEventNumbers) ||
                *eventIter != event) continue;
            indexToEvent = eventIter - eventNumbers().begin() - beginEventNumbers;
          }
          return IndexIntoFileItr(this,
                                  numericalOrder,
                                  kRun,
                                  iRun - runOrLumiIndexes().begin(),
                                  iLumi - runOrLumiIndexes().begin(),
                                  iLumi - runOrLumiIndexes().begin(),
                                  indexToEvent,
                                  endEventNumbers - beginEventNumbers);
        }
      }
    } // Loop over ProcessHistoryIDs

    return IndexIntoFileItr(this,
                            numericalOrder,
                            kEnd,
                            invalidIndex,
                            invalidIndex,
                            invalidIndex,
                            0,
                            0);

  }

  IndexIntoFile::IndexIntoFileItr
  IndexIntoFile::findPosition(SortOrder sortOrder, RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
    if (sortOrder == IndexIntoFile::numericalOrder) {
      return findPosition(run, lumi, event); // a faster algorithm
    }
    IndexIntoFileItr itr = begin(sortOrder);
    IndexIntoFileItr itrEnd = end(sortOrder);
    
    while (itr != itrEnd) {
      if (itr.run() != run) {
        itr.advanceToNextRun();
      }
      else {
        if (lumi == invalidLumi && event == invalidEvent) {
          return itr;
        }
        else if (lumi != invalidLumi && itr.peekAheadAtLumi() != lumi) {
          if (!itr.skipLumiInRun()) {
            itr.advanceToNextRun();
          }
        }
        else {
          if (event == invalidEvent) {
            return itr;
          }
          else {
            EventNumber_t eventNumber = getEventNumberOfEntry(itr.peekAheadAtEventEntry());
            if (eventNumber == event) {
              return itr;
            }
            else {
              if (!itr.skipToNextEventInLumi()) {
                if (!itr.skipLumiInRun()) {
                  itr.advanceToNextRun();
                }
              }
            }
          }
        }
      }
    }
    return itrEnd;
  }

  IndexIntoFile::IndexIntoFileItr
  IndexIntoFile::findEventPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
    assert(event != invalidEvent);
    IndexIntoFileItr iter = findPosition(run, lumi, event);
    iter.advanceToEvent();
    return iter;
  }

  IndexIntoFile::IndexIntoFileItr
  IndexIntoFile::findLumiPosition(RunNumber_t run, LuminosityBlockNumber_t lumi) const {
    assert(lumi != invalidLumi);
    IndexIntoFileItr iter = findPosition(run, lumi, 0U);
    iter.advanceToLumi();
    return iter;
  }

  IndexIntoFile::IndexIntoFileItr
  IndexIntoFile::findRunPosition(RunNumber_t run) const {
    return findPosition(run, 0U, 0U);
  }

  bool
  IndexIntoFile::containsItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
	return event ? containsEvent(run, lumi, event) : (lumi ? containsLumi(run, lumi) : containsRun(run));
  }

  bool
  IndexIntoFile::containsEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
	return findEventPosition(run, lumi, event).getEntryType() != kEnd;
  }

  bool
  IndexIntoFile::containsLumi(RunNumber_t run, LuminosityBlockNumber_t lumi) const {
    return findLumiPosition(run, lumi).getEntryType() != kEnd;
  }

  bool
  IndexIntoFile::containsRun(RunNumber_t run) const {
    return findRunPosition(run).getEntryType() != kEnd;
  }

  IndexIntoFile::SortedRunOrLumiItr IndexIntoFile::beginRunOrLumi() const {
    return SortedRunOrLumiItr(this, 0);
  }

  IndexIntoFile::SortedRunOrLumiItr IndexIntoFile::endRunOrLumi() const {
    return SortedRunOrLumiItr(this, runOrLumiEntries().size());
  }

  void IndexIntoFile::set_intersection(IndexIntoFile const& indexIntoFile,
                                       std::set<IndexRunLumiEventKey> & intersection) const {

    if (empty() || indexIntoFile.empty()) return;
    fillRunOrLumiIndexes();
    indexIntoFile.fillRunOrLumiIndexes();
    RunOrLumiIndexes const& back1 = runOrLumiIndexes().back();
    RunOrLumiIndexes const& back2 = indexIntoFile.runOrLumiIndexes().back();

    // Very quick decision if the run ranges in the two files do not overlap
    if (back2 < runOrLumiIndexes().front()) return;
    if (back1 < indexIntoFile.runOrLumiIndexes().front()) return;

    SortedRunOrLumiItr iter1 = beginRunOrLumi();
    SortedRunOrLumiItr iEnd1 = endRunOrLumi();

    SortedRunOrLumiItr iter2 = indexIntoFile.beginRunOrLumi();
    SortedRunOrLumiItr iEnd2 = indexIntoFile.endRunOrLumi();

    // Quick decision if the lumi ranges in the two files do not overlap
    while (iter1 != iEnd1 && iter1.isRun()) ++iter1;
    if (iter1 == iEnd1) return;
    if (back2 < iter1.runOrLumiIndexes()) return;

    while (iter2 != iEnd2 && iter2.isRun()) ++iter2;
    if (iter2 == iEnd2) return;
    if (back1 < iter2.runOrLumiIndexes()) return;

    RunOrLumiIndexes const* previousIndexes = 0;
 
    // Loop through the both IndexIntoFile objects and look for matching lumis
    while (iter1 != iEnd1 && iter2 != iEnd2) {

      RunOrLumiIndexes const& indexes1 = iter1.runOrLumiIndexes();
      RunOrLumiIndexes const& indexes2 = iter2.runOrLumiIndexes();
      if (indexes1 < indexes2) {
        ++iter1;
      }
      else if (indexes2 < indexes1) {
        ++iter2;
      }
      else { // they are equal

        // Skip them if it is a run or the same lumi
        if (indexes1.isRun() ||
            (previousIndexes && !(*previousIndexes < indexes1))) {
          ++iter1;
          ++iter2;
        }
        else {
          previousIndexes = &indexes1;

          // Found a matching lumi, now look for matching events

          long long beginEventNumbers1 = indexes1.beginEventNumbers();
          long long endEventNumbers1 = indexes1.endEventNumbers();

          long long beginEventNumbers2 = indexes2.beginEventNumbers();
          long long endEventNumbers2 = indexes2.endEventNumbers();

          // there must be at least 1 event in each lumi for there to be any matches
          if ((beginEventNumbers1 >= endEventNumbers1) ||
              (beginEventNumbers2 >= endEventNumbers2)) {
            ++iter1;
            ++iter2;
            continue;
          }

          if (!eventEntries().empty() && !indexIntoFile.eventEntries().empty()) {
	    std::vector<EventEntry> matchingEvents;
            std::insert_iterator<std::vector<EventEntry> > insertIter(matchingEvents, matchingEvents.begin());
	    std::set_intersection(eventEntries().begin() + beginEventNumbers1,
                                  eventEntries().begin() + endEventNumbers1,
                                  indexIntoFile.eventEntries().begin() + beginEventNumbers2,
                                  indexIntoFile.eventEntries().begin() + endEventNumbers2,
                                  insertIter);
            for (std::vector<EventEntry>::const_iterator iEvent = matchingEvents.begin(),
		                                           iEnd = matchingEvents.end();
                 iEvent != iEnd; ++iEvent) {
              intersection.insert(IndexRunLumiEventKey(indexes1.processHistoryIDIndex(),
                                                       indexes1.run(),
                                                       indexes1.lumi(),
                                                       iEvent->event()));
            }
          } else {
            fillEventNumbers();
            indexIntoFile.fillEventNumbers();
	    std::vector<EventNumber_t> matchingEvents;
            std::insert_iterator<std::vector<EventNumber_t> > insertIter(matchingEvents, matchingEvents.begin());
	    std::set_intersection(eventNumbers().begin() + beginEventNumbers1,
                                  eventNumbers().begin() + endEventNumbers1,
                                  indexIntoFile.eventNumbers().begin() + beginEventNumbers2,
                                  indexIntoFile.eventNumbers().begin() + endEventNumbers2,
                                  insertIter);
            for (std::vector<EventNumber_t>::const_iterator iEvent = matchingEvents.begin(),
		                                              iEnd = matchingEvents.end();
                 iEvent != iEnd; ++iEvent) {
              intersection.insert(IndexRunLumiEventKey(indexes1.processHistoryIDIndex(),
                                                       indexes1.run(),
                                                       indexes1.lumi(),
                                                       *iEvent));
            }
          }
        }
      }
    }
  }

  bool IndexIntoFile::containsDuplicateEvents() const {

    RunOrLumiIndexes const* previousIndexes = 0;

    for (SortedRunOrLumiItr iter = beginRunOrLumi(),
                            iEnd = endRunOrLumi();
         iter != iEnd; ++iter) {

      RunOrLumiIndexes const& indexes = iter.runOrLumiIndexes();

      // Skip it if it is a run or the same lumi
      if (indexes.isRun() ||
          (previousIndexes && !(*previousIndexes < indexes))) {
        continue;
      }
      previousIndexes = &indexes;

      long long beginEventNumbers = indexes.beginEventNumbers();
      long long endEventNumbers = indexes.endEventNumbers();

      // there must be more than 1 event in the lumi for there to be any duplicates
      if (beginEventNumbers + 1 >= endEventNumbers) continue;

      if (!eventEntries().empty()) {
        std::vector<EventEntry>::iterator last = eventEntries().begin() + endEventNumbers;
        if (std::adjacent_find(eventEntries().begin() + beginEventNumbers, last) != last) {
          return true;
        }
      } else {
        fillEventNumbers();
        std::vector<EventNumber_t>::iterator last = eventNumbers().begin() + endEventNumbers;
        if (std::adjacent_find(eventNumbers().begin() + beginEventNumbers, last) != last) {
           return true;
        }
      }
    }
    return false;
  }

  IndexIntoFile::RunOrLumiEntry::RunOrLumiEntry() :
    orderPHIDRun_(invalidEntry),
    orderPHIDRunLumi_(invalidEntry),
    entry_(invalidEntry),
    processHistoryIDIndex_(invalidIndex),
    run_(invalidRun),
    lumi_(invalidLumi),
    beginEvents_(invalidEntry),
    endEvents_(invalidEntry) {
  }

  IndexIntoFile::RunOrLumiEntry::RunOrLumiEntry(EntryNumber_t orderPHIDRun,
                                                EntryNumber_t orderPHIDRunLumi,
                                                EntryNumber_t entry,
                                                int processHistoryIDIndex,
                                                RunNumber_t run,
                                                LuminosityBlockNumber_t lumi,
                                                EntryNumber_t beginEvents,
                                                EntryNumber_t endEvents) :
    orderPHIDRun_(orderPHIDRun),
    orderPHIDRunLumi_(orderPHIDRunLumi),
    entry_(entry),
    processHistoryIDIndex_(processHistoryIDIndex),
    run_(run),
    lumi_(lumi),
    beginEvents_(beginEvents),
    endEvents_(endEvents) {
  }

  IndexIntoFile::RunOrLumiIndexes::RunOrLumiIndexes(int processHistoryIDIndex,
                                                    RunNumber_t run,
                                                    LuminosityBlockNumber_t lumi,
                                                    int indexToGetEntry) :
    processHistoryIDIndex_(processHistoryIDIndex),
    run_(run),
    lumi_(lumi),
    indexToGetEntry_(indexToGetEntry),
    beginEventNumbers_(-1),
    endEventNumbers_(-1)
  {
  }

  IndexIntoFile::SortedRunOrLumiItr::SortedRunOrLumiItr(IndexIntoFile const* indexIntoFile, unsigned runOrLumi) :
    indexIntoFile_(indexIntoFile), runOrLumi_(runOrLumi) {
    assert(runOrLumi_ <= indexIntoFile_->runOrLumiEntries().size());
    indexIntoFile_->fillRunOrLumiIndexes();
  }

  bool IndexIntoFile::SortedRunOrLumiItr::operator==(SortedRunOrLumiItr const& right) const {
    return indexIntoFile_ == right.indexIntoFile() &&
           runOrLumi_ == right.runOrLumi();
  }

  bool IndexIntoFile::SortedRunOrLumiItr::operator!=(SortedRunOrLumiItr const& right) const {
    return indexIntoFile_ != right.indexIntoFile() ||
           runOrLumi_ != right.runOrLumi();
  }

  IndexIntoFile::SortedRunOrLumiItr & IndexIntoFile::SortedRunOrLumiItr::operator++() {
    if (runOrLumi_ != indexIntoFile_->runOrLumiEntries().size()) {
      ++runOrLumi_;
    }
    return *this;
  }

  bool IndexIntoFile::SortedRunOrLumiItr::isRun() {
    return indexIntoFile_->runOrLumiIndexes().at(runOrLumi_).lumi() == invalidLumi;
  }

  void IndexIntoFile::SortedRunOrLumiItr::getRange(long long & beginEventNumbers,
                long long & endEventNumbers,
                EntryNumber_t & beginEventEntry,
                EntryNumber_t & endEventEntry) {
    beginEventNumbers = indexIntoFile_->runOrLumiIndexes().at(runOrLumi_).beginEventNumbers();
    endEventNumbers = indexIntoFile_->runOrLumiIndexes().at(runOrLumi_).endEventNumbers();

    int indexToGetEntry = indexIntoFile_->runOrLumiIndexes().at(runOrLumi_).indexToGetEntry();
    beginEventEntry = indexIntoFile_->runOrLumiEntries_.at(indexToGetEntry).beginEvents();
    endEventEntry = indexIntoFile_->runOrLumiEntries_.at(indexToGetEntry).endEvents();
  }

  IndexIntoFile::RunOrLumiIndexes const&
  IndexIntoFile::SortedRunOrLumiItr::runOrLumiIndexes() const {
    return indexIntoFile_->runOrLumiIndexes().at(runOrLumi_);
  }

  IndexIntoFile::IndexIntoFileItrImpl::IndexIntoFileItrImpl(IndexIntoFile const* indexIntoFile,
                       EntryType entryType,
                       int indexToRun,
                       int indexToLumi,
                       int indexToEventRange,
                       long long indexToEvent,
                       long long nEvents) :
    indexIntoFile_(indexIntoFile),
    size_(static_cast<int>(indexIntoFile_->runOrLumiEntries_.size())),
    type_(entryType),
    indexToRun_(indexToRun),
    indexToLumi_(indexToLumi),
    indexToEventRange_(indexToEventRange),
    indexToEvent_(indexToEvent),
    nEvents_(nEvents) {
  }

   IndexIntoFile::IndexIntoFileItrImpl::~IndexIntoFileItrImpl() {}
   
  void IndexIntoFile::IndexIntoFileItrImpl::next() {

    if (type_ == kEvent) {
      if ((indexToEvent_ + 1)  < nEvents_) {
        ++indexToEvent_;
      }
      else {
        bool found = nextEventRange();

        if (!found) {
          type_ = getRunOrLumiEntryType(indexToLumi_ + 1);

          if (type_ == kLumi) {
            ++indexToLumi_;
            initializeLumi();
          }
          else if (type_ == kRun) {
            indexToRun_ = indexToLumi_ + 1;
            initializeRun();
          }
          else {
            setInvalid(); // type_ is kEnd
          }
        }
      }
    }
    else if (type_ == kLumi) {

      if (indexToLumi_ + 1 == size_) {
        if (indexToEvent_ < nEvents_) {
          type_ = kEvent;
        }
        else {
          setInvalid();
        }
      }
      else {

        EntryType nextType = getRunOrLumiEntryType(indexToLumi_ + 1);

        if (nextType == kLumi && isSameLumi(indexToLumi_, indexToLumi_ + 1)) {
          ++indexToLumi_;
        }
        else if (indexToEvent_ < nEvents_) {
          type_ = kEvent;
        }
        else if (nextType == kRun) {
          type_ = kRun;
          indexToRun_ = indexToLumi_ + 1;
          initializeRun();
        }
        else {
          ++indexToLumi_;
          initializeLumi();
        }
      }
    }
    else if (type_ == kRun) {
      EntryType nextType = getRunOrLumiEntryType(indexToRun_ + 1);
      bool sameRun = isSameRun(indexToRun_, indexToRun_ + 1);
      if (nextType == kRun && sameRun) {
        ++indexToRun_;
      }
      else if (nextType == kRun && !sameRun) {
        ++indexToRun_;
        initializeRun();
      }
      else if (nextType == kLumi) {
        type_ = kLumi;
      }
      else {
        setInvalid();
      }
    }
  }

  void IndexIntoFile::IndexIntoFileItrImpl::skipEventForward(int & phIndexOfSkippedEvent,
                                                             RunNumber_t & runOfSkippedEvent,
                                                             LuminosityBlockNumber_t & lumiOfSkippedEvent,
                                                             EntryNumber_t & skippedEventEntry) {
    if (indexToEvent_  < nEvents_) {
      phIndexOfSkippedEvent = processHistoryIDIndex();
      runOfSkippedEvent = run();
      lumiOfSkippedEvent = peekAheadAtLumi();
      skippedEventEntry = peekAheadAtEventEntry();

      if ((indexToEvent_ + 1)  < nEvents_) {
        ++indexToEvent_;
        return;
      }
      else if (nextEventRange()) {
        return;
      }
      else if (type_ == kRun || type_ == kLumi) {
        if (skipLumiInRun()) {
          return;
        }
      }
      else if (type_ == kEvent) {
        next();
        return;
      }
      advanceToNextRun();
      return;
    }

    if (type_ == kRun) {
      while (skipLumiInRun()) {
        if (indexToEvent_  < nEvents_) {
          skipEventForward(phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, skippedEventEntry);
          return;
        }
      }
    }

    while (indexToEvent_ >= nEvents_ && type_ != kEnd) {
      next();
    }
    if (type_ == kEnd) {
      phIndexOfSkippedEvent = invalidIndex;
      runOfSkippedEvent = invalidRun;
      lumiOfSkippedEvent = invalidLumi;
      skippedEventEntry = invalidEntry;
      return;
    }
    skipEventForward(phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, skippedEventEntry);
    return;
  }

  void IndexIntoFile::IndexIntoFileItrImpl::skipEventBackward(int& phIndexOfEvent,
                                                              RunNumber_t& runOfEvent,
                                                              LuminosityBlockNumber_t& lumiOfEvent,
                                                              EntryNumber_t& eventEntry) {
    // Look for previous events in the current lumi
    if (indexToEvent_ > 0) {
      --indexToEvent_;
    }
    else if (!previousEventRange()) {

      // Look for previous events in previous lumis
      if (!previousLumiWithEvents()) {

        // If we get here there are no previous events in the file

        if (!indexIntoFile_->empty()) {
	    // Set the iterator to the beginning of the file  
	    type_ = kRun;
            indexToRun_ = 0;
            initializeRun();
	}
        phIndexOfEvent = invalidIndex;
        runOfEvent = invalidRun;
        lumiOfEvent = invalidLumi;
        eventEntry = invalidEntry;
        return;
      }
    }
    // Found a previous event and we have set the iterator so that this event
    // will be the next event process. (There may or may not be a run and/or
    // a lumi processed first).
    // Return information about this event
    phIndexOfEvent = processHistoryIDIndex();
    runOfEvent = run();
    lumiOfEvent = peekAheadAtLumi();
    eventEntry = peekAheadAtEventEntry();
  }

  bool IndexIntoFile::IndexIntoFileItrImpl::previousLumiWithEvents() {
    // Find the correct place to start the search
    int newLumi = indexToLumi();
    if (newLumi == invalidIndex) {
      newLumi = indexToRun() == invalidIndex ? size() - 1 : indexToRun();
    }
    else {
      while (getRunOrLumiEntryType(newLumi - 1) == kLumi &&
             isSameLumi(newLumi, newLumi - 1)) {
        --newLumi;
      }
      --newLumi;
    }
    if (newLumi <= 0) return false;

    // Look backwards for a lumi with events
    for ( ; newLumi > 0; --newLumi) {
      if (getRunOrLumiEntryType(newLumi) == kRun) {
	continue;
      }
      if (setToLastEventInRange(newLumi)) {
        break;  // found it
      }
    }
    if (newLumi == 0) return false;

    // Finish initializing the iterator
    while (getRunOrLumiEntryType(newLumi - 1) == kLumi &&
           isSameLumi(newLumi, newLumi - 1)) {
      --newLumi;
    }
    setIndexToLumi(newLumi);

    if (type() != kEnd &&
        isSameRun(newLumi, indexToRun())) {
      if (type() == kEvent) type_ = kLumi;
      return true;
    }
    int newRun = newLumi;
    while (newRun > 0 && getRunOrLumiEntryType(newRun - 1) == kLumi) {
      --newRun;
    }
    --newRun;
    assert(getRunOrLumiEntryType(newRun) == kRun);
    while (getRunOrLumiEntryType(newRun - 1) == kRun &&
           isSameRun(newRun - 1, newLumi)) {
      --newRun;
    }
    indexToRun_ = newRun;
    type_ = kRun;
    return true;
  }

  void IndexIntoFile::IndexIntoFileItrImpl::advanceToNextRun() {
    if (type_ == kEnd) return;
    for (int i = 1; indexToRun_ + i < size_; ++i) {
      if (getRunOrLumiEntryType(indexToRun_ + i) == kRun) {
	if (!isSameRun(indexToRun_, indexToRun_ + i)) {
          type_ = kRun;
          indexToRun_ += i;
          initializeRun();
          return;
        }
      }
    }
    setInvalid();
  }

  void IndexIntoFile::IndexIntoFileItrImpl::advanceToNextLumiOrRun() {
    if (type_ == kEnd) return;
    assert(indexToRun_ != invalidIndex);

    // A preliminary step is to advance to the last run entry for
    // this run (actually this step is not needed in the
    // context I expect this to be called in, just being careful)
    int startSearch = indexToRun_;    
    for (int i = 1; startSearch + i < size_; ++i) {
      if (getRunOrLumiEntryType(startSearch + i) == kRun && 
          isSameRun(indexToRun_, startSearch + i)) {
	indexToRun_ = startSearch + i;
      }
      else {
        break;
      }
    }

    if (type_ == kRun && indexToLumi_ != invalidIndex) {
      type_ = kLumi;
      return;
    }

    startSearch = indexToLumi_;
    if (startSearch == invalidIndex) startSearch = indexToRun_;
    for (int i = 1; startSearch + i < size_; ++i) {
      if (getRunOrLumiEntryType(startSearch + i) == kRun) {
        if (!isSameRun(indexToRun_, startSearch + i)) {
          type_ = kRun;
          indexToRun_ = startSearch + i;
          initializeRun();
          return;
        }
      }
      else if (indexToLumi_ != invalidIndex) {
        if (!isSameLumi(indexToLumi_, startSearch + i)) {
          type_ = kLumi;
          indexToLumi_ = startSearch + i;
          initializeLumi();
          return;
        }
      }
    }
    setInvalid();
  }

  bool IndexIntoFile::IndexIntoFileItrImpl::skipToNextEventInLumi() {
    if (indexToEvent_ >= nEvents_) return false;
    if ((indexToEvent_ + 1)  < nEvents_) {
      ++indexToEvent_;
      return true;
    }
    return nextEventRange();
  }

  void IndexIntoFile::IndexIntoFileItrImpl::initializeRun() {

    indexToLumi_ = invalidIndex;
    indexToEventRange_ = invalidIndex;
    indexToEvent_ = 0;
    nEvents_ = 0;

    for (int i = 1; (i + indexToRun_) < size_; ++i) {
      EntryType entryType = getRunOrLumiEntryType(indexToRun_ + i);
      bool sameRun = isSameRun(indexToRun_, indexToRun_ + i);

      if (entryType == kRun) {
        if (sameRun) {
          continue;
        }
        else {
          break;
        }
      }
      else {
        indexToLumi_ = indexToRun_ + i;
        initializeLumi();
        return;
      }
    }
  }

  bool IndexIntoFile::IndexIntoFileItrImpl::operator==(IndexIntoFileItrImpl const& right) const {
    return (indexIntoFile_ == right.indexIntoFile_ &&
            size_ == right.size_ &&
            type_ == right.type_ &&
            indexToRun_ == right.indexToRun_ &&
            indexToLumi_ == right.indexToLumi_ &&
            indexToEventRange_ == right.indexToEventRange_ &&
            indexToEvent_ == right.indexToEvent_ &&
            nEvents_ == right.nEvents_);
  }

  void
  IndexIntoFile::IndexIntoFileItrImpl::copyPosition(IndexIntoFileItrImpl const& position) {
    type_ = position.type_;
    indexToRun_ = position.indexToRun_;
    indexToLumi_ = position.indexToLumi_;
    indexToEventRange_ = position.indexToEventRange_;
    indexToEvent_ = position.indexToEvent_;
    nEvents_ = position.nEvents_;
  }

  void IndexIntoFile::IndexIntoFileItrImpl::setInvalid() {
    type_ = kEnd;
    indexToRun_ = invalidIndex;
    indexToLumi_ = invalidIndex;
    indexToEventRange_ = invalidIndex;
    indexToEvent_ = 0;
    nEvents_ = 0;
  }

  IndexIntoFile::IndexIntoFileItrNoSort::IndexIntoFileItrNoSort(IndexIntoFile const* indexIntoFile,
                         EntryType entryType,
                         int indexToRun,
                         int indexToLumi,
                         int indexToEventRange,
                         long long indexToEvent,
                         long long nEvents) :
    IndexIntoFileItrImpl(indexIntoFile,
                         entryType,
                         indexToRun,
                         indexToLumi,
                         indexToEventRange,
                         indexToEvent,
                         nEvents)
  {
  }

  IndexIntoFile::IndexIntoFileItrImpl*
  IndexIntoFile::IndexIntoFileItrNoSort::clone() const {
    return new IndexIntoFileItrNoSort(*this);
  }

  int
  IndexIntoFile::IndexIntoFileItrNoSort::processHistoryIDIndex() const {
    if (type() == kEnd) return invalidIndex;
    return indexIntoFile()->runOrLumiEntries()[indexToRun()].processHistoryIDIndex();
  }

  RunNumber_t IndexIntoFile::IndexIntoFileItrNoSort::run() const {
    if (type() == kEnd) return invalidRun;
    return indexIntoFile()->runOrLumiEntries()[indexToRun()].run();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrNoSort::lumi() const {
    if (type() == kEnd || type() == kRun) return invalidLumi;
    return indexIntoFile()->runOrLumiEntries()[indexToLumi()].lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrNoSort::entry() const {
    if (type() == kEnd) return invalidEntry;
    if (type() == kRun) return indexIntoFile()->runOrLumiEntries()[indexToRun()].entry();
    if (type() == kLumi) return indexIntoFile()->runOrLumiEntries()[indexToLumi()].entry();
    return
      indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents() +
      indexToEvent();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrNoSort::peekAheadAtLumi() const {
    if (indexToLumi() == invalidIndex) return invalidLumi;
    return indexIntoFile()->runOrLumiEntries()[indexToLumi()].lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrNoSort::peekAheadAtEventEntry() const {
    if (indexToLumi() == invalidIndex) return invalidEntry;
    if (indexToEvent() >= nEvents()) return invalidEntry;
    return
      indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents() +
      indexToEvent();
  }

  void IndexIntoFile::IndexIntoFileItrNoSort::initializeLumi_() {
    assert(indexToLumi() != invalidIndex);

    setIndexToEventRange(invalidIndex);
    setIndexToEvent(0);
    setNEvents(0);

    for (int i = 0; indexToLumi() + i < size(); ++i) {
      if (indexIntoFile()->runOrLumiEntries()[indexToLumi() + i].isRun()) {
        break;
      }
      else if (indexIntoFile()->runOrLumiEntries()[indexToLumi() + i].lumi() ==
               indexIntoFile()->runOrLumiEntries()[indexToLumi()].lumi()) {
        if (indexIntoFile()->runOrLumiEntries()[indexToLumi() + i].beginEvents() == invalidEntry) {
          continue;
        }
        setIndexToEventRange(indexToLumi() + i);
        setIndexToEvent(0);
        setNEvents(indexIntoFile()->runOrLumiEntries()[indexToEventRange()].endEvents() -
		   indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents());
        break;
      }
      else {
        break;
      }
    }
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::nextEventRange() {
    if (indexToEventRange() == invalidIndex) return false;

    // Look for the next event range, same lumi but different entry 
    for(int i = 1; indexToEventRange() + i < size(); ++i) {
      if (indexIntoFile()->runOrLumiEntries()[indexToEventRange() + i ].isRun()) {
        return false;  // hit next run
      }
      else if (indexIntoFile()->runOrLumiEntries()[indexToEventRange() + i].lumi() ==
               indexIntoFile()->runOrLumiEntries()[indexToEventRange()].lumi()) {
        if (indexIntoFile()->runOrLumiEntries()[indexToEventRange() + i].beginEvents() == invalidEntry) {
          continue; // same lumi but has no events, keep looking
        }
        setIndexToEventRange(indexToEventRange() + i);
        setIndexToEvent(0);
        setNEvents(indexIntoFile()->runOrLumiEntries()[indexToEventRange()].endEvents() -
                   indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents());
        return true; // found more events in this lumi
      }
      return false; // hit next lumi
    }
    return false; // hit the end of the IndexIntoFile
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::previousEventRange() {
    if (indexToEventRange() == invalidIndex) return false;
    assert(indexToEventRange() < size());

    // Look backward for a previous event range with events, same lumi but different entry 
    for(int i = 1; indexToEventRange() - i > 0; ++i) {
      int newRange = indexToEventRange() - i;
      if (indexIntoFile()->runOrLumiEntries()[newRange].isRun()) {
        return false;  // hit run
      }
      else if (isSameLumi(newRange, indexToEventRange())) {
        if (indexIntoFile()->runOrLumiEntries()[newRange].beginEvents() == invalidEntry) {
          continue; // same lumi but has no events, keep looking
        }
        setIndexToEventRange(newRange);
        setNEvents(indexIntoFile()->runOrLumiEntries()[indexToEventRange()].endEvents() -
                   indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents());
        setIndexToEvent(nEvents() - 1);
        return true; // found previous event in this lumi
      }
      return false; // hit previous lumi
    }
    return false; // hit the beginning of the IndexIntoFile, 0th entry has to be a run
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::setToLastEventInRange(int index) {
    if (indexIntoFile()->runOrLumiEntries()[index].beginEvents() == invalidEntry) {
      return false;
    }
    setIndexToEventRange(index);
    setNEvents(indexIntoFile()->runOrLumiEntries()[indexToEventRange()].endEvents() -
               indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents());
    assert(nEvents() > 0);
    setIndexToEvent(nEvents() - 1);
    return true;
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::skipLumiInRun() { 
    if (indexToLumi() == invalidIndex) return false;
    for(int i = 1; indexToLumi() + i < size(); ++i) {
      int newLumi = indexToLumi() + i;
      if (indexIntoFile()->runOrLumiEntries()[newLumi].isRun()) {
        return false;  // hit next run
      }
      else if (indexIntoFile()->runOrLumiEntries()[newLumi].lumi() ==
               indexIntoFile()->runOrLumiEntries()[indexToLumi()].lumi()) {
        continue;
      }
      setIndexToLumi(newLumi);
      initializeLumi();
      return true; // hit next lumi
    }
    return false; // hit the end of the IndexIntoFile
  }

  IndexIntoFile::EntryType IndexIntoFile::IndexIntoFileItrNoSort::getRunOrLumiEntryType(int index) const {
    if (index < 0 || index >= size()) {
      return kEnd;
    }
    else if (indexIntoFile()->runOrLumiEntries()[index].isRun()) {
      return kRun;
    }
    return kLumi;
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::isSameLumi(int index1, int index2) const {
    if (index1 < 0 || index1 >= size() || index2 < 0 || index2 >= size()) {
      return false;
    }
    return indexIntoFile()->runOrLumiEntries()[index1].lumi() ==
           indexIntoFile()->runOrLumiEntries()[index2].lumi();
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::isSameRun(int index1, int index2) const {
    if (index1 < 0 || index1 >= size() || index2 < 0 || index2 >= size()) {
      return false;
    }
    return indexIntoFile()->runOrLumiEntries()[index1].run() ==
           indexIntoFile()->runOrLumiEntries()[index2].run() &&
           indexIntoFile()->runOrLumiEntries()[index1].processHistoryIDIndex() ==
           indexIntoFile()->runOrLumiEntries()[index2].processHistoryIDIndex();
  }

  IndexIntoFile::IndexIntoFileItrSorted::IndexIntoFileItrSorted(IndexIntoFile const* indexIntoFile,
                         EntryType entryType,
                         int indexToRun,
                         int indexToLumi,
                         int indexToEventRange,
                         long long indexToEvent,
                         long long nEvents) :
    IndexIntoFileItrImpl(indexIntoFile,
                         entryType,
                         indexToRun,
                         indexToLumi,
                         indexToEventRange,
                         indexToEvent,
                         nEvents) {
    indexIntoFile->fillRunOrLumiIndexes();
  }

  IndexIntoFile::IndexIntoFileItrImpl* IndexIntoFile::IndexIntoFileItrSorted::clone() const {
    return new IndexIntoFileItrSorted(*this);
  }

  int IndexIntoFile::IndexIntoFileItrSorted::processHistoryIDIndex() const {
    if (type() == kEnd) return invalidIndex;
    return indexIntoFile()->runOrLumiIndexes()[indexToRun()].processHistoryIDIndex();
  }

  RunNumber_t IndexIntoFile::IndexIntoFileItrSorted::run() const {
    if (type() == kEnd) return invalidRun;
    return indexIntoFile()->runOrLumiIndexes()[indexToRun()].run();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrSorted::lumi() const {
    if (type() == kEnd || type() == kRun) return invalidLumi;
    return indexIntoFile()->runOrLumiIndexes()[indexToLumi()].lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrSorted::entry() const {
    if (type() == kEnd) return invalidEntry;
    if (type() == kRun) {
      int i =  indexIntoFile()->runOrLumiIndexes()[indexToRun()].indexToGetEntry();
      return indexIntoFile()->runOrLumiEntries()[i].entry();
    }
    if (type() == kLumi) {
      int i =  indexIntoFile()->runOrLumiIndexes()[indexToLumi()].indexToGetEntry();
      return indexIntoFile()->runOrLumiEntries()[i].entry();
    }
    long long eventNumberIndex = 
      indexIntoFile()->runOrLumiIndexes()[indexToEventRange()].beginEventNumbers() +
      indexToEvent();
    indexIntoFile()->fillEventEntries();
    return indexIntoFile()->eventEntries().at(eventNumberIndex).entry();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrSorted::peekAheadAtLumi() const {
    if (indexToLumi() == invalidIndex) return invalidLumi;
    return indexIntoFile()->runOrLumiIndexes()[indexToLumi()].lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrSorted::peekAheadAtEventEntry() const {
    if (indexToLumi() == invalidIndex) return invalidEntry;
    if (indexToEvent() >= nEvents()) return invalidEntry;
    long long eventNumberIndex = 
      indexIntoFile()->runOrLumiIndexes()[indexToEventRange()].beginEventNumbers() +
      indexToEvent();
    indexIntoFile()->fillEventEntries();
    return indexIntoFile()->eventEntries().at(eventNumberIndex).entry();
  }

  void IndexIntoFile::IndexIntoFileItrSorted::initializeLumi_() {
    assert(indexToLumi() != invalidIndex);
    setIndexToEventRange(indexToLumi());
    setIndexToEvent(0);
    setNEvents( 
      indexIntoFile()->runOrLumiIndexes()[indexToLumi()].endEventNumbers() -
      indexIntoFile()->runOrLumiIndexes()[indexToLumi()].beginEventNumbers());
    if (nEvents() == 0){
      setIndexToEventRange(invalidIndex);
    }
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::nextEventRange() {
    return false;          
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::previousEventRange() {
    return false;          
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::setToLastEventInRange(int index) {
    long long nEventsInRange = 
      indexIntoFile()->runOrLumiIndexes()[index].endEventNumbers() -
      indexIntoFile()->runOrLumiIndexes()[index].beginEventNumbers();
    if (nEventsInRange == 0) {
      return false;
    }
    while (index > 0 &&
           !indexIntoFile()->runOrLumiIndexes()[index - 1].isRun() &&
           isSameLumi(index, index - 1)) {
      --index;
    }
    assert(nEventsInRange ==
      indexIntoFile()->runOrLumiIndexes()[index].endEventNumbers() -
      indexIntoFile()->runOrLumiIndexes()[index].beginEventNumbers());

    setIndexToEventRange(index);
    setNEvents(nEventsInRange);
    assert(nEvents() > 0);
    setIndexToEvent(nEventsInRange - 1);
    return true;
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::skipLumiInRun() { 
    if (indexToLumi() == invalidIndex) return false;
    for(int i = 1; indexToLumi() + i < size(); ++i) {
      int newLumi = indexToLumi() + i;
      if (indexIntoFile()->runOrLumiIndexes()[newLumi].isRun()) {
        return false;  // hit next run
      }
      else if (indexIntoFile()->runOrLumiIndexes()[newLumi].lumi() ==
               indexIntoFile()->runOrLumiIndexes()[indexToLumi()].lumi()) {
        continue;
      }
      setIndexToLumi(newLumi);
      initializeLumi();
      return true; // hit next lumi
    }
    return false; // hit the end of the IndexIntoFile
  }

  IndexIntoFile::EntryType IndexIntoFile::IndexIntoFileItrSorted::getRunOrLumiEntryType(int index) const {
    if (index < 0 || index >= size()) {
      return kEnd;
    }
    else if (indexIntoFile()->runOrLumiIndexes()[index].isRun()) {
      return kRun;
    }
    return kLumi;
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::isSameLumi(int index1, int index2) const {
    if (index1 < 0 || index1 >= size() || index2 < 0 || index2 >= size()) {
      return false;
    }
    return indexIntoFile()->runOrLumiIndexes()[index1].lumi() ==
           indexIntoFile()->runOrLumiIndexes()[index2].lumi();
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::isSameRun(int index1, int index2) const {
    if (index1 < 0 || index1 >= size() || index2 < 0 || index2 >= size()) {
      return false;
    }
    return indexIntoFile()->runOrLumiIndexes()[index1].run() ==
           indexIntoFile()->runOrLumiIndexes()[index2].run() &&
           indexIntoFile()->runOrLumiIndexes()[index1].processHistoryIDIndex() ==
           indexIntoFile()->runOrLumiIndexes()[index2].processHistoryIDIndex();
  }

  IndexIntoFile::IndexIntoFileItr::IndexIntoFileItr(IndexIntoFile const* indexIntoFile,
                   SortOrder sortOrder,
                   EntryType entryType,
                   int indexToRun,
                   int indexToLumi,
                   int indexToEventRange,
                   long long indexToEvent,
                   long long nEvents) :
    impl_() {
    if (sortOrder == numericalOrder) {
      value_ptr<IndexIntoFileItrImpl> temp(new IndexIntoFileItrSorted(indexIntoFile,
                                                                      entryType,
                                                                      indexToRun,
                                                                      indexToLumi,
                                                                      indexToEventRange,
                                                                      indexToEvent,
                                                                      nEvents
                                                                     ));
      swap(temp, impl_);
    }
    else {
      value_ptr<IndexIntoFileItrImpl> temp(new IndexIntoFileItrNoSort(indexIntoFile,
                                                                      entryType,
                                                                      indexToRun,
                                                                      indexToLumi,
                                                                      indexToEventRange,
                                                                      indexToEvent,
                                                                      nEvents));
      swap(temp, impl_);
    }
  }

  void IndexIntoFile::IndexIntoFileItr::advanceToEvent() {
    for (EntryType entryType = getEntryType();
         entryType != kEnd && entryType != kEvent;
         entryType = getEntryType()) {
	    impl_->next();
    }
  }

  void IndexIntoFile::IndexIntoFileItr::advanceToLumi() {
    for (EntryType entryType = getEntryType();
         entryType != kEnd && entryType != kLumi;
         entryType = getEntryType()) {
	    impl_->next();
    }
  }

  void
  IndexIntoFile::IndexIntoFileItr::copyPosition(IndexIntoFileItr const& position) {
    impl_->copyPosition(*position.impl_);
  }

  IndexIntoFile::Transients::Transients() : previousAddedIndex_(invalidIndex),
                                            runToFirstEntry_(),
                                            lumiToFirstEntry_(),
                                            beginEvents_(invalidEntry),
                                            endEvents_(invalidEntry),
                                            currentIndex_(invalidIndex),
                                            currentRun_(invalidRun),
                                            currentLumi_(invalidLumi),
                                            numberOfEvents_(0),
                                            eventFinder_(),
                                            runOrLumiIndexes_(),
                                            eventNumbers_(),
                                            eventEntries_(),
                                            unsortedEventNumbers_() {
  }

  bool Compare_Index_Run::operator()(IndexIntoFile::RunOrLumiIndexes const& lh, IndexIntoFile::RunOrLumiIndexes const& rh) {
    if (lh.processHistoryIDIndex() == rh.processHistoryIDIndex()) {
      return lh.run() < rh.run();
    }
    return lh.processHistoryIDIndex() < rh.processHistoryIDIndex();
  }

  bool Compare_Index::operator()(IndexIntoFile::RunOrLumiIndexes const& lh, IndexIntoFile::RunOrLumiIndexes const& rh) {
    return lh.processHistoryIDIndex() < rh.processHistoryIDIndex();
  }
}
