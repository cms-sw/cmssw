#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <iomanip>
#include <ostream>

namespace edm {

  int const IndexIntoFile::invalidIndex;
  RunNumber_t const IndexIntoFile::invalidRun;
  LuminosityBlockNumber_t const IndexIntoFile::invalidLumi;
  EventNumber_t const IndexIntoFile::invalidEvent;
  IndexIntoFile::EntryNumber_t const IndexIntoFile::invalidEntry;

  IndexIntoFile::Transients::Transients()
      : previousAddedIndex_(invalidIndex),
        runToOrder_(),
        lumiToOrder_(),
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
        unsortedEventNumbers_() {}

  void IndexIntoFile::Transients::reset() {
    previousAddedIndex_ = invalidIndex;
    runToOrder_.clear();
    lumiToOrder_.clear();
    beginEvents_ = invalidEntry;
    endEvents_ = invalidEntry;
    currentIndex_ = invalidIndex;
    currentRun_ = invalidRun;
    currentLumi_ = invalidLumi;
    numberOfEvents_ = 0;
    eventFinder_ = nullptr;  // propagate_const<T> has no reset() function
    runOrLumiIndexes_.clear();
    eventNumbers_.clear();
    eventEntries_.clear();
    unsortedEventNumbers_.clear();
  }

  IndexIntoFile::IndexIntoFile() : transient_(), processHistoryIDs_(), runOrLumiEntries_() {}

  IndexIntoFile::~IndexIntoFile() {}

  ProcessHistoryID const& IndexIntoFile::processHistoryID(int i) const { return processHistoryIDs_.at(i); }

  std::vector<ProcessHistoryID> const& IndexIntoFile::processHistoryIDs() const { return processHistoryIDs_; }

  void IndexIntoFile::addLumi(int index, RunNumber_t run, LuminosityBlockNumber_t lumi, EntryNumber_t entry) {
    // assign each lumi an order value sequentially when first seen
    std::pair<IndexRunLumiKey, EntryNumber_t> keyAndOrder{IndexRunLumiKey{index, run, lumi}, lumiToOrder().size()};
    lumiToOrder().insert(keyAndOrder);  // does nothing if this key already was inserted
    runOrLumiEntries_.emplace_back(invalidEntry,
                                   lumiToOrder()[IndexRunLumiKey{index, run, lumi}],
                                   entry,
                                   index,
                                   run,
                                   lumi,
                                   beginEvents(),
                                   endEvents());
    beginEvents() = invalidEntry;
    endEvents() = invalidEntry;
  }

  void IndexIntoFile::addEntry(ProcessHistoryID const& processHistoryID,
                               RunNumber_t run,
                               LuminosityBlockNumber_t lumi,
                               EventNumber_t event,
                               EntryNumber_t entry) {
    int index = 0;
    // First see if the ProcessHistoryID is the same as the previous one.
    // This is just a performance optimization.  We expect to usually get
    // many in a row that are the same.
    if (previousAddedIndex() != invalidIndex && processHistoryID == processHistoryIDs_[previousAddedIndex()]) {
      index = previousAddedIndex();
    } else {
      // If it was not the same as the previous one then search through the
      // entire vector.  If it is not there, it needs to be added at the
      // end.
      index = 0;
      while (index < static_cast<int>(processHistoryIDs_.size()) && processHistoryIDs_[index] != processHistoryID) {
        ++index;
      }
      if (index == static_cast<int>(processHistoryIDs_.size())) {
        processHistoryIDs_.push_back(processHistoryID);
      }
    }
    previousAddedIndex() = index;

    if (lumi == invalidLumi) {  // adding a run entry
      std::pair<IndexRunKey, EntryNumber_t> keyAndOrder{IndexRunKey{index, run}, runToOrder().size()};
      runToOrder().insert(keyAndOrder);  // does nothing if this key already was inserted
      runOrLumiEntries_.emplace_back(
          runToOrder()[IndexRunKey{index, run}], invalidEntry, entry, index, run, lumi, invalidEntry, invalidEntry);
    } else {
      if (event == invalidEvent) {  // adding a lumi entry
        if ((currentIndex() != index or currentRun() != run or currentLumi() != lumi) and
            currentLumi() != invalidLumi) {
          //we have overlapping lumis so must inject a placeholder
          addLumi(currentIndex(), currentRun(), currentLumi(), invalidEntry);
        }
        addLumi(index, run, lumi, entry);
        currentIndex() = invalidIndex;
        currentRun() = invalidRun;
        currentLumi() = invalidLumi;
        std::pair<IndexRunKey, EntryNumber_t> keyAndOrder{IndexRunKey{index, run}, runToOrder().size()};
        runToOrder().insert(keyAndOrder);  // does nothing if this key already was inserted
      } else {                             // adding an event entry
        if ((currentIndex() != index or currentRun() != run or currentLumi() != lumi) and
            currentLumi() != invalidLumi) {
          //We have overlapping lumis so need to inject a placeholder
          addLumi(currentIndex(), currentRun(), currentLumi(), invalidEntry);
        }
        setNumberOfEvents(numberOfEvents() + 1);
        if (beginEvents() == invalidEntry) {
          currentRun() = run;
          currentIndex() = index;
          currentLumi() = lumi;
          beginEvents() = entry;
          endEvents() = beginEvents() + 1;
          std::pair<IndexRunKey, EntryNumber_t> keyAndOrder{IndexRunKey{index, run}, runToOrder().size()};
          runToOrder().insert(keyAndOrder);  // does nothing if this key already was inserted
        } else {
          assert(currentIndex() == index);
          assert(currentRun() == run);
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
    for (RunOrLumiEntry const& item : runOrLumiEntries_) {
      runOrLumiIndexes().emplace_back(item.processHistoryIDIndex(), item.run(), item.lumi(), index);
      ++index;
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
      if (beginOfLumi == iEnd)
        break;

      endOfLumi = beginOfLumi + 1;
      while (endOfLumi != iEnd && beginOfLumi->processHistoryIDIndex() == endOfLumi->processHistoryIDIndex() &&
             beginOfLumi->run() == endOfLumi->run() && beginOfLumi->lumi() == endOfLumi->lumi()) {
        ++endOfLumi;
      }
      int nEvents = 0;
      for (std::vector<RunOrLumiIndexes>::iterator iter = beginOfLumi; iter != endOfLumi; ++iter) {
        if (runOrLumiEntries_[iter->indexToGetEntry()].beginEvents() != invalidEntry) {
          nEvents += runOrLumiEntries_[iter->indexToGetEntry()].endEvents() -
                     runOrLumiEntries_[iter->indexToGetEntry()].beginEvents();
        }
      }
      for (std::vector<RunOrLumiIndexes>::iterator iter = beginOfLumi; iter != endOfLumi; ++iter) {
        iter->setBeginEventNumbers(beginEventNumbers);
        iter->setEndEventNumbers(beginEventNumbers + nEvents);
      }
      beginEventNumbers += nEvents;
      beginOfLumi = endOfLumi;
    }
    assert(runOrLumiIndexes().size() == runOrLumiEntries_.size());
  }

  void IndexIntoFile::fillEventNumbers() const { fillEventNumbersOrEntries(true, false); }

  void IndexIntoFile::fillEventEntries() const { fillEventNumbersOrEntries(false, true); }

  void IndexIntoFile::fillEventNumbersOrEntries(bool needEventNumbers, bool needEventEntries) const {
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

    for (SortedRunOrLumiItr runOrLumi = beginRunOrLumi(), runOrLumiEnd = endRunOrLumi(); runOrLumi != runOrLumiEnd;
         ++runOrLumi) {
      if (runOrLumi.isRun())
        continue;

      long long beginEventNumbers = 0;
      long long endEventNumbers = 0;
      EntryNumber_t beginEventEntry = invalidEntry;
      EntryNumber_t endEventEntry = invalidEntry;
      runOrLumi.getRange(beginEventNumbers, endEventNumbers, beginEventEntry, endEventEntry);

      // This is true each time one hits a new lumi section (except if the previous lumi had
      // no events, in which case the offset is still 0 anyway)
      if (beginEventNumbers != previousBeginEventNumbers)
        offset = 0;

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

  void IndexIntoFile::fillUnsortedEventNumbers() const {
    if (numberOfEvents() == 0 || !unsortedEventNumbers().empty()) {
      return;
    }
    unsortedEventNumbersMutable().reserve(numberOfEvents());

    // The main purpose for the existence of the unsortedEventNumbers
    // vector is that it can easily be filled by reading through
    // the EventAuxiliary branch in the same order as the TTree
    // entries. fillEventNumbersOrEntries can then use this information
    // instead of using getEventNumberOfEntry directly and reading
    // the branch in a different order.
    for (std::vector<EventNumber_t>::size_type entry = 0U; entry < numberOfEvents(); ++entry) {
      unsortedEventNumbersMutable().push_back(getEventNumberOfEntry(entry));
    }
  }

  // We are closing the input file, but we need to keep event numbers.
  // We can delete the other transient collections by using the swap trick.

  void IndexIntoFile::inputFileClosed() {
    std::vector<EventEntry>().swap(eventEntries());
    std::vector<RunOrLumiIndexes>().swap(runOrLumiIndexes());
    std::vector<EventNumber_t>().swap(unsortedEventNumbers());
    resetEventFinder();
  }

  void IndexIntoFile::doneFileInitialization() { std::vector<EventNumber_t>().swap(unsortedEventNumbers()); }

  void IndexIntoFile::reduceProcessHistoryIDs(ProcessHistoryRegistry const& processHistoryRegistry) {
    std::vector<ProcessHistoryID> reducedPHIDs;

    std::map<ProcessHistoryID, int> reducedPHIDToIndex;
    std::pair<ProcessHistoryID, int> mapEntry(ProcessHistoryID(), 0);
    std::pair<std::map<ProcessHistoryID, int>::iterator, bool> insertResult;

    std::vector<int> phidIndexConverter;
    for (auto const& phid : processHistoryIDs_) {
      ProcessHistoryID const& reducedPHID = processHistoryRegistry.reducedProcessHistoryID(phid);
      mapEntry.first = reducedPHID;
      insertResult = reducedPHIDToIndex.insert(mapEntry);

      if (insertResult.second) {
        insertResult.first->second = reducedPHIDs.size();
        reducedPHIDs.push_back(reducedPHID);
      }
      phidIndexConverter.push_back(insertResult.first->second);
    }
    processHistoryIDs_.swap(reducedPHIDs);

    // If the size of the vector of IDs does not change
    // then their indexes and the ordering of the Runs and
    // and Lumis does not change, so we are done.
    if (processHistoryIDs_.size() == reducedPHIDs.size()) {
      return;
    }

    std::map<IndexIntoFile::IndexRunKey, int> runOrderMap;
    std::pair<std::map<IndexIntoFile::IndexRunKey, int>::iterator, bool> runInsertResult;

    std::map<IndexIntoFile::IndexRunLumiKey, int> lumiOrderMap;
    std::pair<std::map<IndexIntoFile::IndexRunLumiKey, int>::iterator, bool> lumiInsertResult;

    // loop over all the RunOrLumiEntry's
    for (auto& item : runOrLumiEntries_) {
      // Convert the process history index so it points into the new vector of reduced IDs
      item.setProcessHistoryIDIndex(phidIndexConverter.at(item.processHistoryIDIndex()));

      // Convert the phid-run order
      IndexIntoFile::IndexRunKey runKey(item.processHistoryIDIndex(), item.run());
      runInsertResult = runOrderMap.insert(std::pair<IndexIntoFile::IndexRunKey, int>(runKey, 0));
      if (runInsertResult.second) {
        runInsertResult.first->second = item.orderPHIDRun();
      } else {
        item.setOrderPHIDRun(runInsertResult.first->second);
      }

      // Convert the phid-run-lumi order for the lumi entries
      if (item.lumi() != 0) {
        IndexIntoFile::IndexRunLumiKey lumiKey(item.processHistoryIDIndex(), item.run(), item.lumi());
        lumiInsertResult = lumiOrderMap.insert(std::pair<IndexIntoFile::IndexRunLumiKey, int>(lumiKey, 0));
        if (lumiInsertResult.second) {
          lumiInsertResult.first->second = item.orderPHIDRunLumi();
        } else {
          item.setOrderPHIDRunLumi(lumiInsertResult.first->second);
        }
      }
    }
    std::stable_sort(runOrLumiEntries_.begin(), runOrLumiEntries_.end());
  }

  void IndexIntoFile::fixIndexes(std::vector<ProcessHistoryID>& processHistoryIDs) {
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
      } else {
        oldToNewIndex[iter - processHistoryIDs_.begin()] = iterExisting - processHistoryIDs.begin();
      }
    }
    processHistoryIDs_ = processHistoryIDs;

    for (RunOrLumiEntry& item : runOrLumiEntries_) {
      item.setProcessHistoryIDIndex(oldToNewIndex[item.processHistoryIDIndex()]);
    }
  }

  void IndexIntoFile::sortVector_Run_Or_Lumi_Entries() {
    for (RunOrLumiEntry& item : runOrLumiEntries_) {
      std::map<IndexRunKey, EntryNumber_t>::const_iterator keyAndOrder =
          runToOrder().find(IndexRunKey(item.processHistoryIDIndex(), item.run()));
      if (keyAndOrder == runToOrder().end()) {
        throw Exception(errors::LogicError)
            << "In IndexIntoFile::sortVector_Run_Or_Lumi_Entries. A run entry is missing.\n"
            << "This means the IndexIntoFile product in the output file will be corrupted.\n"
            << "The output file will be unusable for most purposes.\n"
            << "If this occurs after an unrelated exception was thrown in\n"
            << "endLuminosityBlock or endRun then ignore this exception and fix\n"
            << "the primary exception. This is an expected side effect.\n"
            << "Otherwise please report this to the core framework developers\n";
      }
      item.setOrderPHIDRun(keyAndOrder->second);
    }
    stable_sort_all(runOrLumiEntries_);
    checkForMissingRunOrLumiEntry();
  }

  void IndexIntoFile::checkForMissingRunOrLumiEntry() const {
    bool shouldThrow = false;
    bool foundValidLumiEntry = true;
    EntryNumber_t currentRun = invalidEntry;
    EntryNumber_t currentLumi = invalidEntry;
    EntryNumber_t previousLumi = invalidEntry;

    RunOrLumiEntry const* lastEntry = nullptr;
    for (RunOrLumiEntry const& item : runOrLumiEntries_) {
      if (item.isRun()) {
        currentRun = item.orderPHIDRun();
      } else {  // Lumi
        if (item.orderPHIDRun() != currentRun) {
          throw Exception(errors::LogicError)
              << "In IndexIntoFile::sortVector_Run_Or_Lumi_Entries. Missing Run entry.\n"
              << "If this occurs after an unrelated exception occurs, please ignore this\n"
              << "exception and fix the primary exception. This is a possible and expected\n"
              << "side effect. Otherwise, please report to Framework developers.\n"
              << "This could indicate a bug in the source or Framework\n"
              << "Run: " << item.run() << " Lumi: " << item.lumi() << " Entry: " << item.entry() << "\n";
        }
        currentLumi = item.orderPHIDRunLumi();
        if (currentLumi != previousLumi) {
          if (!foundValidLumiEntry) {
            shouldThrow = true;
            break;
          }
          foundValidLumiEntry = false;
          previousLumi = currentLumi;
        }
        if (item.entry() != invalidEntry) {
          foundValidLumiEntry = true;
        }
      }
      lastEntry = &item;
    }
    if (!foundValidLumiEntry) {
      shouldThrow = true;
    }

    if (shouldThrow) {
      throw Exception(errors::LogicError)
          << "In IndexIntoFile::sortVector_Run_Or_Lumi_Entries. Missing valid Lumi entry.\n"
          << "If this occurs after an unrelated exception occurs, please ignore this\n"
          << "exception and fix the primary exception. This is a possible and expected\n"
          << "side effect. Otherwise, please report to Framework developers.\n"
          << "This could indicate a bug in the source or Framework\n"
          << "Run: " << lastEntry->run() << " Lumi: " << lastEntry->lumi() << " Entry: " << lastEntry->entry() << "\n";
    }
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
      if (beginOfLumi == iEnd)
        break;

      endOfLumi = beginOfLumi + 1;
      while (endOfLumi != iEnd && beginOfLumi->processHistoryIDIndex() == endOfLumi->processHistoryIDIndex() &&
             beginOfLumi->run() == endOfLumi->run() && beginOfLumi->lumi() == endOfLumi->lumi()) {
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
      if (beginOfLumi == iEnd)
        break;

      endOfLumi = beginOfLumi + 1;
      while (endOfLumi != iEnd && beginOfLumi->processHistoryIDIndex() == endOfLumi->processHistoryIDIndex() &&
             beginOfLumi->run() == endOfLumi->run() && beginOfLumi->lumi() == endOfLumi->lumi()) {
        ++endOfLumi;
      }
      assert(beginOfLumi->endEventNumbers() >= 0);
      assert(beginOfLumi->endEventNumbers() <= static_cast<long long>(eventEntries().size()));
      std::sort(eventEntries().begin() + beginOfLumi->beginEventNumbers(),
                eventEntries().begin() + beginOfLumi->endEventNumbers());
      beginOfLumi = endOfLumi;
    }
  }

  IndexIntoFile::IndexIntoFileItr IndexIntoFile::begin(SortOrder sortOrder) const {
    if (empty()) {
      return end(sortOrder);
    }
    IndexIntoFileItr iter(this, sortOrder, kRun, 0, invalidIndex, invalidIndex, 0, 0);
    iter.initializeRun();
    return iter;
  }

  IndexIntoFile::IndexIntoFileItr IndexIntoFile::end(SortOrder sortOrder) const {
    return IndexIntoFileItr(this, sortOrder, kEnd, invalidIndex, invalidIndex, invalidIndex, 0, 0);
  }

  bool IndexIntoFile::iterationWillBeInEntryOrder(SortOrder sortOrder) const {
    EntryNumber_t maxEntry = invalidEntry;
    for (IndexIntoFileItr it = begin(sortOrder), itEnd = end(sortOrder); it != itEnd; ++it) {
      if (it.getEntryType() == kEvent) {
        if (it.entry() < maxEntry) {
          return false;
        }
        maxEntry = it.entry();
      }
    }
    return true;
  }

  bool IndexIntoFile::empty() const { return runOrLumiEntries().empty(); }

  IndexIntoFile::IndexIntoFileItr IndexIntoFile::findPosition(RunNumber_t run,
                                                              LuminosityBlockNumber_t lumi,
                                                              EventNumber_t event) const {
    fillRunOrLumiIndexes();

    bool lumiMissing = (lumi == 0 && event != 0);

    std::vector<RunOrLumiIndexes>::const_iterator iEnd = runOrLumiIndexes().end();
    std::vector<RunOrLumiIndexes>::const_iterator phEnd;

    // Loop over ranges of entries with the same ProcessHistoryID
    for (std::vector<RunOrLumiIndexes>::const_iterator phBegin = runOrLumiIndexes().begin(); phBegin != iEnd;
         phBegin = phEnd) {
      RunOrLumiIndexes el(phBegin->processHistoryIDIndex(), run, lumi, 0);
      phEnd = std::upper_bound(phBegin, iEnd, el, Compare_Index());

      std::vector<RunOrLumiIndexes>::const_iterator iRun = std::lower_bound(phBegin, phEnd, el, Compare_Index_Run());

      if (iRun == phEnd || iRun->run() != run)
        continue;

      if (lumi == invalidLumi && event == invalidEvent) {
        IndexIntoFileItr indexItr(
            this, numericalOrder, kRun, iRun - runOrLumiIndexes().begin(), invalidIndex, invalidIndex, 0, 0);
        indexItr.initializeRun();
        return indexItr;
      }

      std::vector<RunOrLumiIndexes>::const_iterator iRunEnd = std::upper_bound(iRun, phEnd, el, Compare_Index_Run());
      if (!lumiMissing) {
        std::vector<RunOrLumiIndexes>::const_iterator iLumi = std::lower_bound(iRun, iRunEnd, el);
        if (iLumi == iRunEnd || iLumi->lumi() != lumi)
          continue;

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
        if (beginEventNumbers >= endEventNumbers)
          continue;

        long long indexToEvent = 0;
        if (!eventEntries().empty()) {
          std::vector<EventEntry>::const_iterator eventIter =
              std::lower_bound(eventEntries().begin() + beginEventNumbers,
                               eventEntries().begin() + endEventNumbers,
                               EventEntry(event, invalidEntry));
          if (eventIter == (eventEntries().begin() + endEventNumbers) || eventIter->event() != event)
            continue;

          indexToEvent = eventIter - eventEntries().begin() - beginEventNumbers;
        } else {
          fillEventNumbers();
          std::vector<EventNumber_t>::const_iterator eventIter = std::lower_bound(
              eventNumbers().begin() + beginEventNumbers, eventNumbers().begin() + endEventNumbers, event);
          if (eventIter == (eventNumbers().begin() + endEventNumbers) || *eventIter != event)
            continue;

          indexToEvent = eventIter - eventNumbers().begin() - beginEventNumbers;
        }

        int newIndexToLumi = iLumi - runOrLumiIndexes().begin();
        while (runOrLumiEntries_[runOrLumiIndexes()[newIndexToLumi].indexToGetEntry()].entry() == invalidEntry) {
          ++newIndexToLumi;
          assert(static_cast<unsigned>(newIndexToLumi) < runOrLumiEntries_.size());
          assert(runOrLumiIndexes()[newIndexToLumi].lumi() == lumi);
        }

        return IndexIntoFileItr(this,
                                numericalOrder,
                                kRun,
                                iRun - runOrLumiIndexes().begin(),
                                newIndexToLumi,
                                iLumi - runOrLumiIndexes().begin(),
                                indexToEvent,
                                endEventNumbers - beginEventNumbers);
      }
      if (lumiMissing) {
        std::vector<RunOrLumiIndexes>::const_iterator iLumi = iRun;
        while (iLumi != iRunEnd && iLumi->lumi() == invalidLumi) {
          ++iLumi;
        }
        if (iLumi == iRunEnd)
          continue;

        std::vector<RunOrLumiIndexes>::const_iterator lumiEnd;
        for (; iLumi != iRunEnd; iLumi = lumiEnd) {
          RunOrLumiIndexes elWithLumi(phBegin->processHistoryIDIndex(), run, iLumi->lumi(), 0);
          lumiEnd = std::upper_bound(iLumi, iRunEnd, elWithLumi);

          long long beginEventNumbers = iLumi->beginEventNumbers();
          long long endEventNumbers = iLumi->endEventNumbers();
          if (beginEventNumbers >= endEventNumbers)
            continue;

          long long indexToEvent = 0;
          if (!eventEntries().empty()) {
            std::vector<EventEntry>::const_iterator eventIter =
                std::lower_bound(eventEntries().begin() + beginEventNumbers,
                                 eventEntries().begin() + endEventNumbers,
                                 EventEntry(event, invalidEntry));
            if (eventIter == (eventEntries().begin() + endEventNumbers) || eventIter->event() != event)
              continue;
            indexToEvent = eventIter - eventEntries().begin() - beginEventNumbers;
          } else {
            fillEventNumbers();
            std::vector<EventNumber_t>::const_iterator eventIter = std::lower_bound(
                eventNumbers().begin() + beginEventNumbers, eventNumbers().begin() + endEventNumbers, event);
            if (eventIter == (eventNumbers().begin() + endEventNumbers) || *eventIter != event)
              continue;
            indexToEvent = eventIter - eventNumbers().begin() - beginEventNumbers;
          }

          int newIndexToLumi = iLumi - runOrLumiIndexes().begin();
          while (runOrLumiEntries_[runOrLumiIndexes()[newIndexToLumi].indexToGetEntry()].entry() == invalidEntry) {
            ++newIndexToLumi;
            assert(static_cast<unsigned>(newIndexToLumi) < runOrLumiEntries_.size());
            assert(runOrLumiIndexes()[newIndexToLumi].lumi() == iLumi->lumi());
          }

          return IndexIntoFileItr(this,
                                  numericalOrder,
                                  kRun,
                                  iRun - runOrLumiIndexes().begin(),
                                  newIndexToLumi,
                                  iLumi - runOrLumiIndexes().begin(),
                                  indexToEvent,
                                  endEventNumbers - beginEventNumbers);
        }
      }
    }  // Loop over ProcessHistoryIDs

    return IndexIntoFileItr(this, numericalOrder, kEnd, invalidIndex, invalidIndex, invalidIndex, 0, 0);
  }

  IndexIntoFile::IndexIntoFileItr IndexIntoFile::findPosition(SortOrder sortOrder,
                                                              RunNumber_t run,
                                                              LuminosityBlockNumber_t lumi,
                                                              EventNumber_t event) const {
    if (sortOrder == IndexIntoFile::numericalOrder) {
      return findPosition(run, lumi, event);  // a faster algorithm
    }
    IndexIntoFileItr itr = begin(sortOrder);
    IndexIntoFileItr itrEnd = end(sortOrder);

    while (itr != itrEnd) {
      if (itr.run() != run) {
        itr.advanceToNextRun();
      } else {
        if (lumi == invalidLumi && event == invalidEvent) {
          return itr;
        } else if (lumi != invalidLumi && itr.peekAheadAtLumi() != lumi) {
          if (!itr.skipLumiInRun()) {
            itr.advanceToNextRun();
          }
        } else {
          if (event == invalidEvent) {
            return itr;
          } else {
            EventNumber_t eventNumber = getEventNumberOfEntry(itr.peekAheadAtEventEntry());
            if (eventNumber == event) {
              return itr;
            } else {
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

  IndexIntoFile::IndexIntoFileItr IndexIntoFile::findEventPosition(RunNumber_t run,
                                                                   LuminosityBlockNumber_t lumi,
                                                                   EventNumber_t event) const {
    assert(event != invalidEvent);
    IndexIntoFileItr iter = findPosition(run, lumi, event);
    iter.advanceToEvent();
    return iter;
  }

  IndexIntoFile::IndexIntoFileItr IndexIntoFile::findLumiPosition(RunNumber_t run, LuminosityBlockNumber_t lumi) const {
    assert(lumi != invalidLumi);
    IndexIntoFileItr iter = findPosition(run, lumi, 0U);
    iter.advanceToLumi();
    return iter;
  }

  IndexIntoFile::IndexIntoFileItr IndexIntoFile::findRunPosition(RunNumber_t run) const {
    return findPosition(run, 0U, 0U);
  }

  bool IndexIntoFile::containsItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
    return (event != 0) ? containsEvent(run, lumi, event) : (lumi ? containsLumi(run, lumi) : containsRun(run));
  }

  bool IndexIntoFile::containsEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
    return findEventPosition(run, lumi, event).getEntryType() != kEnd;
  }

  bool IndexIntoFile::containsLumi(RunNumber_t run, LuminosityBlockNumber_t lumi) const {
    return findLumiPosition(run, lumi).getEntryType() != kEnd;
  }

  bool IndexIntoFile::containsRun(RunNumber_t run) const { return findRunPosition(run).getEntryType() != kEnd; }

  IndexIntoFile::SortedRunOrLumiItr IndexIntoFile::beginRunOrLumi() const { return SortedRunOrLumiItr(this, 0); }

  IndexIntoFile::SortedRunOrLumiItr IndexIntoFile::endRunOrLumi() const {
    return SortedRunOrLumiItr(this, runOrLumiEntries().size());
  }

  void IndexIntoFile::set_intersection(IndexIntoFile const& indexIntoFile,
                                       std::set<IndexRunLumiEventKey>& intersection) const {
    if (empty() || indexIntoFile.empty())
      return;
    fillRunOrLumiIndexes();
    indexIntoFile.fillRunOrLumiIndexes();
    RunOrLumiIndexes const& back1 = runOrLumiIndexes().back();
    RunOrLumiIndexes const& back2 = indexIntoFile.runOrLumiIndexes().back();

    // Very quick decision if the run ranges in the two files do not overlap
    if (back2 < runOrLumiIndexes().front())
      return;
    if (back1 < indexIntoFile.runOrLumiIndexes().front())
      return;

    SortedRunOrLumiItr iter1 = beginRunOrLumi();
    SortedRunOrLumiItr iEnd1 = endRunOrLumi();

    SortedRunOrLumiItr iter2 = indexIntoFile.beginRunOrLumi();
    SortedRunOrLumiItr iEnd2 = indexIntoFile.endRunOrLumi();

    // Quick decision if the lumi ranges in the two files do not overlap
    while (iter1 != iEnd1 && iter1.isRun())
      ++iter1;
    if (iter1 == iEnd1)
      return;
    if (back2 < iter1.runOrLumiIndexes())
      return;

    while (iter2 != iEnd2 && iter2.isRun())
      ++iter2;
    if (iter2 == iEnd2)
      return;
    if (back1 < iter2.runOrLumiIndexes())
      return;

    RunOrLumiIndexes const* previousIndexes = nullptr;

    // Loop through the both IndexIntoFile objects and look for matching lumis
    while (iter1 != iEnd1 && iter2 != iEnd2) {
      RunOrLumiIndexes const& indexes1 = iter1.runOrLumiIndexes();
      RunOrLumiIndexes const& indexes2 = iter2.runOrLumiIndexes();
      if (indexes1 < indexes2) {
        ++iter1;
      } else if (indexes2 < indexes1) {
        ++iter2;
      } else {  // they are equal

        // Skip them if it is a run or the same lumi
        if (indexes1.isRun() || (previousIndexes && !(*previousIndexes < indexes1))) {
          ++iter1;
          ++iter2;
        } else {
          previousIndexes = &indexes1;

          // Found a matching lumi, now look for matching events

          long long beginEventNumbers1 = indexes1.beginEventNumbers();
          long long endEventNumbers1 = indexes1.endEventNumbers();

          long long beginEventNumbers2 = indexes2.beginEventNumbers();
          long long endEventNumbers2 = indexes2.endEventNumbers();

          // there must be at least 1 event in each lumi for there to be any matches
          if ((beginEventNumbers1 >= endEventNumbers1) || (beginEventNumbers2 >= endEventNumbers2)) {
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
            for (EventEntry const& entry : matchingEvents) {
              intersection.insert(IndexRunLumiEventKey(
                  indexes1.processHistoryIDIndex(), indexes1.run(), indexes1.lumi(), entry.event()));
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
            for (EventNumber_t const& eventNumber : matchingEvents) {
              intersection.insert(
                  IndexRunLumiEventKey(indexes1.processHistoryIDIndex(), indexes1.run(), indexes1.lumi(), eventNumber));
            }
          }
        }
      }
    }
  }

  bool IndexIntoFile::containsDuplicateEvents() const {
    RunOrLumiIndexes const* previousIndexes = nullptr;

    for (SortedRunOrLumiItr iter = beginRunOrLumi(), iEnd = endRunOrLumi(); iter != iEnd; ++iter) {
      RunOrLumiIndexes const& indexes = iter.runOrLumiIndexes();

      // Skip it if it is a run or the same lumi
      if (indexes.isRun() || (previousIndexes && !(*previousIndexes < indexes))) {
        continue;
      }
      previousIndexes = &indexes;

      long long beginEventNumbers = indexes.beginEventNumbers();
      long long endEventNumbers = indexes.endEventNumbers();

      // there must be more than 1 event in the lumi for there to be any duplicates
      if (beginEventNumbers + 1 >= endEventNumbers)
        continue;

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

  IndexIntoFile::RunOrLumiEntry::RunOrLumiEntry()
      : orderPHIDRun_(invalidEntry),
        orderPHIDRunLumi_(invalidEntry),
        entry_(invalidEntry),
        processHistoryIDIndex_(invalidIndex),
        run_(invalidRun),
        lumi_(invalidLumi),
        beginEvents_(invalidEntry),
        endEvents_(invalidEntry) {}

  IndexIntoFile::RunOrLumiEntry::RunOrLumiEntry(EntryNumber_t orderPHIDRun,
                                                EntryNumber_t orderPHIDRunLumi,
                                                EntryNumber_t entry,
                                                int processHistoryIDIndex,
                                                RunNumber_t run,
                                                LuminosityBlockNumber_t lumi,
                                                EntryNumber_t beginEvents,
                                                EntryNumber_t endEvents)
      : orderPHIDRun_(orderPHIDRun),
        orderPHIDRunLumi_(orderPHIDRunLumi),
        entry_(entry),
        processHistoryIDIndex_(processHistoryIDIndex),
        run_(run),
        lumi_(lumi),
        beginEvents_(beginEvents),
        endEvents_(endEvents) {}

  IndexIntoFile::RunOrLumiIndexes::RunOrLumiIndexes(int processHistoryIDIndex,
                                                    RunNumber_t run,
                                                    LuminosityBlockNumber_t lumi,
                                                    int indexToGetEntry)
      : processHistoryIDIndex_(processHistoryIDIndex),
        run_(run),
        lumi_(lumi),
        indexToGetEntry_(indexToGetEntry),
        beginEventNumbers_(-1),
        endEventNumbers_(-1) {}

  IndexIntoFile::SortedRunOrLumiItr::SortedRunOrLumiItr(IndexIntoFile const* indexIntoFile, unsigned runOrLumi)
      : indexIntoFile_(indexIntoFile), runOrLumi_(runOrLumi) {
    assert(runOrLumi_ <= indexIntoFile_->runOrLumiEntries().size());
    indexIntoFile_->fillRunOrLumiIndexes();
  }

  bool IndexIntoFile::SortedRunOrLumiItr::operator==(SortedRunOrLumiItr const& right) const {
    return indexIntoFile_ == right.indexIntoFile() && runOrLumi_ == right.runOrLumi();
  }

  bool IndexIntoFile::SortedRunOrLumiItr::operator!=(SortedRunOrLumiItr const& right) const {
    return indexIntoFile_ != right.indexIntoFile() || runOrLumi_ != right.runOrLumi();
  }

  IndexIntoFile::SortedRunOrLumiItr& IndexIntoFile::SortedRunOrLumiItr::operator++() {
    if (runOrLumi_ != indexIntoFile_->runOrLumiEntries().size()) {
      ++runOrLumi_;
    }
    return *this;
  }

  bool IndexIntoFile::SortedRunOrLumiItr::isRun() {
    return indexIntoFile_->runOrLumiIndexes().at(runOrLumi_).lumi() == invalidLumi;
  }

  void IndexIntoFile::SortedRunOrLumiItr::getRange(long long& beginEventNumbers,
                                                   long long& endEventNumbers,
                                                   EntryNumber_t& beginEventEntry,
                                                   EntryNumber_t& endEventEntry) {
    beginEventNumbers = indexIntoFile_->runOrLumiIndexes().at(runOrLumi_).beginEventNumbers();
    endEventNumbers = indexIntoFile_->runOrLumiIndexes().at(runOrLumi_).endEventNumbers();

    int indexToGetEntry = indexIntoFile_->runOrLumiIndexes().at(runOrLumi_).indexToGetEntry();
    beginEventEntry = indexIntoFile_->runOrLumiEntries_.at(indexToGetEntry).beginEvents();
    endEventEntry = indexIntoFile_->runOrLumiEntries_.at(indexToGetEntry).endEvents();
  }

  IndexIntoFile::RunOrLumiIndexes const& IndexIntoFile::SortedRunOrLumiItr::runOrLumiIndexes() const {
    return indexIntoFile_->runOrLumiIndexes().at(runOrLumi_);
  }

  IndexIntoFile::IndexIntoFileItrImpl::IndexIntoFileItrImpl(IndexIntoFile const* indexIntoFile,
                                                            EntryType entryType,
                                                            int indexToRun,
                                                            int indexToLumi,
                                                            int indexToEventRange,
                                                            long long indexToEvent,
                                                            long long nEvents)
      : indexIntoFile_(indexIntoFile),
        size_(static_cast<int>(indexIntoFile_->runOrLumiEntries_.size())),
        type_(entryType),
        indexToRun_(indexToRun),
        indexToLumi_(indexToLumi),
        indexToEventRange_(indexToEventRange),
        indexToEvent_(indexToEvent),
        nEvents_(nEvents) {}

  IndexIntoFile::IndexIntoFileItrImpl::~IndexIntoFileItrImpl() {}

  void IndexIntoFile::IndexIntoFileItrImpl::next() {
    if (type_ == kEvent) {
      if ((indexToEvent_ + 1) < nEvents_) {
        ++indexToEvent_;
      } else {
        bool found = nextEventRange();

        if (!found) {
          type_ = getRunOrLumiEntryType(indexToLumi_ + 1);

          if (type_ == kLumi) {
            ++indexToLumi_;
            initializeLumi();
          } else if (type_ == kRun) {
            indexToRun_ = indexToLumi_ + 1;
            initializeRun();
          } else {
            setInvalid();  // type_ is kEnd
          }
        }
      }
    } else if (type_ == kLumi) {
      if (indexToLumi_ + 1 == indexedSize()) {
        if (indexToEvent_ < nEvents_) {
          type_ = kEvent;
        } else {
          setInvalid();
        }
      } else {
        EntryType nextType = getRunOrLumiEntryType(indexToLumi_ + 1);

        if (nextType == kLumi && isSameLumi(indexToLumi_, indexToLumi_ + 1)) {
          ++indexToLumi_;
        } else if (indexToEvent_ < nEvents_) {
          type_ = kEvent;
        } else if (nextType == kRun) {
          type_ = kRun;
          indexToRun_ = indexToLumi_ + 1;
          initializeRun();
        } else {
          ++indexToLumi_;
          initializeLumi();
        }
      }
    } else if (type_ == kRun) {
      EntryType nextType = getRunOrLumiEntryType(indexToRun_ + 1);
      bool sameRun = isSameRun(indexToRun_, indexToRun_ + 1);
      if (nextType == kRun && sameRun) {
        ++indexToRun_;
      } else if (nextType == kRun && !sameRun) {
        ++indexToRun_;
        initializeRun();
      } else if (nextType == kLumi) {
        type_ = kLumi;
      } else {
        setInvalid();
      }
    }
  }

  void IndexIntoFile::IndexIntoFileItrImpl::skipEventForward(int& phIndexOfSkippedEvent,
                                                             RunNumber_t& runOfSkippedEvent,
                                                             LuminosityBlockNumber_t& lumiOfSkippedEvent,
                                                             EntryNumber_t& skippedEventEntry) {
    if (indexToEvent_ < nEvents_) {
      phIndexOfSkippedEvent = processHistoryIDIndex();
      runOfSkippedEvent = run();
      lumiOfSkippedEvent = peekAheadAtLumi();
      skippedEventEntry = peekAheadAtEventEntry();

      if ((indexToEvent_ + 1) < nEvents_) {
        ++indexToEvent_;
        return;
      } else if (nextEventRange()) {
        return;
      } else if (type_ == kRun || type_ == kLumi) {
        if (skipLumiInRun()) {
          return;
        }
      } else if (type_ == kEvent) {
        next();
        return;
      }
      advanceToNextRun();
      return;
    }

    if (type_ == kRun) {
      while (skipLumiInRun()) {
        if (indexToEvent_ < nEvents_) {
          skipEventForward(phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, skippedEventEntry);
          return;
        }
      }
    }

    while (indexToEvent_ >= nEvents_ && type_ != kEnd) {
      while (skipLumiInRun()) {
        if (indexToEvent_ < nEvents_) {
          skipEventForward(phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, skippedEventEntry);
          return;
        }
      }
      advanceToNextRun();
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
    } else if (!previousEventRange()) {
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
      newLumi = indexToRun() == invalidIndex ? indexedSize() - 1 : indexToRun();
    } else {
      while (getRunOrLumiEntryType(newLumi - 1) == kLumi && isSameLumi(newLumi, newLumi - 1)) {
        --newLumi;
      }
      --newLumi;
    }
    if (newLumi <= 0)
      return false;

    // Look backwards for a lumi with events
    for (; newLumi > 0; --newLumi) {
      if (getRunOrLumiEntryType(newLumi) == kRun) {
        continue;
      }
      if (setToLastEventInRange(newLumi)) {
        break;  // found it
      }
    }
    if (newLumi == 0)
      return false;

    // Finish initializing the iterator
    // Go back to the first lumi entry for this lumi
    while (getRunOrLumiEntryType(newLumi - 1) == kLumi && isSameLumi(newLumi, newLumi - 1)) {
      --newLumi;
    }
    // Then go forward to the first valid one (or if there are not any valid ones
    // to the last one, only possible in the entryOrder case)
    while (not lumiIterationStartingIndex(newLumi)) {
      ++newLumi;
    }

    setIndexToLumi(newLumi);

    if (type() != kEnd && isSameRun(newLumi, indexToRun())) {
      if (type() == kEvent)
        type_ = kLumi;
      return true;
    }
    int newRun = newLumi;
    while (newRun > 0 && getRunOrLumiEntryType(newRun - 1) == kLumi) {
      --newRun;
    }
    --newRun;
    assert(getRunOrLumiEntryType(newRun) == kRun);
    while (getRunOrLumiEntryType(newRun - 1) == kRun && isSameRun(newRun - 1, newLumi)) {
      --newRun;
    }
    indexToRun_ = newRun;
    type_ = kRun;
    return true;
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrImpl::firstEventEntryThisRun() {
    if (indexToLumi() == invalidIndex)
      return invalidEntry;

    int saveIndexToLumi = indexToLumi();
    int saveIndexToEventRange = indexToEventRange();
    long long saveIndexToEvent = indexToEvent();
    long long saveNEvents = nEvents();

    initializeRun();

    IndexIntoFile::EntryNumber_t returnValue = invalidEntry;

    do {
      if (indexToEvent() < nEvents()) {
        returnValue = peekAheadAtEventEntry();
        break;
      }
    } while (skipLumiInRun());

    setIndexToLumi(saveIndexToLumi);
    setIndexToEventRange(saveIndexToEventRange);
    setIndexToEvent(saveIndexToEvent);
    setNEvents(saveNEvents);

    return returnValue;
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrImpl::firstEventEntryThisLumi() {
    if (indexToLumi() == invalidIndex)
      return invalidEntry;

    int saveIndexToLumi = indexToLumi();
    int saveIndexToEventRange = indexToEventRange();
    long long saveIndexToEvent = indexToEvent();
    long long saveNEvents = nEvents();

    while (indexToLumi() - 1 > 0) {
      if (getRunOrLumiEntryType(indexToLumi() - 1) == kRun)
        break;
      if (!isSameLumi(indexToLumi(), indexToLumi() - 1))
        break;
      --indexToLumi_;
    }
    initializeLumi();

    IndexIntoFile::EntryNumber_t returnValue = invalidEntry;

    if (indexToEvent() < nEvents()) {
      returnValue = peekAheadAtEventEntry();
    }

    setIndexToLumi(saveIndexToLumi);
    setIndexToEventRange(saveIndexToEventRange);
    setIndexToEvent(saveIndexToEvent);
    setNEvents(saveNEvents);

    return returnValue;
  }

  void IndexIntoFile::IndexIntoFileItrImpl::advanceToNextRun() {
    if (type_ == kEnd)
      return;
    for (int i = 1; indexToRun_ + i < indexedSize(); ++i) {
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
    if (type_ == kEnd)
      return;
    assert(indexToRun_ != invalidIndex);

    // A preliminary step is to advance to the last run entry for
    // this run (actually this step is not needed in the
    // context I expect this to be called in, just being careful)
    int startSearch = indexToRun_;
    for (int i = 1; startSearch + i < indexedSize(); ++i) {
      if (getRunOrLumiEntryType(startSearch + i) == kRun && isSameRun(indexToRun_, startSearch + i)) {
        indexToRun_ = startSearch + i;
      } else {
        break;
      }
    }

    if (type_ == kRun && indexToLumi_ != invalidIndex) {
      type_ = kLumi;
      return;
    }

    startSearch = indexToLumi_;
    if (startSearch == invalidIndex)
      startSearch = indexToRun_;
    for (int i = 1; startSearch + i < indexedSize(); ++i) {
      if (getRunOrLumiEntryType(startSearch + i) == kRun) {
        if (!isSameRun(indexToRun_, startSearch + i)) {
          type_ = kRun;
          indexToRun_ = startSearch + i;
          initializeRun();
          return;
        }
      } else if (indexToLumi_ != invalidIndex) {
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
    if (indexToEvent_ >= nEvents_)
      return false;
    if ((indexToEvent_ + 1) < nEvents_) {
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

    for (int i = indexToRun_ + 1, iEnd = indexedSize(); i < iEnd; ++i) {
      EntryType entryType = getRunOrLumiEntryType(i);
      bool sameRun = isSameRun(indexToRun_, i);

      if (entryType == kRun) {
        if (sameRun) {
          continue;
        } else {
          break;
        }
      } else {
        indexToLumi_ = i;
        initializeLumi();
        return;
      }
    }
  }

  void IndexIntoFile::IndexIntoFileItrImpl::initializeLumi() {
    initializeLumi_();
    auto oldLumi = lumi();
    // Then go forward to the first valid one (or if there are not any valid ones
    // to the last one, only possible in the entryOrder case)
    while (not lumiIterationStartingIndex(indexToLumi_)) {
      ++indexToLumi_;
    }
    assert(oldLumi == lumi());
  }

  bool IndexIntoFile::IndexIntoFileItrImpl::operator==(IndexIntoFileItrImpl const& right) const {
    return (indexIntoFile_ == right.indexIntoFile_ && size_ == right.size_ && type_ == right.type_ &&
            indexToRun_ == right.indexToRun_ && indexToLumi_ == right.indexToLumi_ &&
            indexToEventRange_ == right.indexToEventRange_ && indexToEvent_ == right.indexToEvent_ &&
            nEvents_ == right.nEvents_);
  }

  int IndexIntoFile::IndexIntoFileItrImpl::indexedSize() const { return size(); }

  void IndexIntoFile::IndexIntoFileItrImpl::copyPosition(IndexIntoFileItrImpl const& position) {
    type_ = position.type_;
    indexToRun_ = position.indexToRun_;
    indexToLumi_ = position.indexToLumi_;
    indexToEventRange_ = position.indexToEventRange_;
    indexToEvent_ = position.indexToEvent_;
    nEvents_ = position.nEvents_;
  }

  void IndexIntoFile::IndexIntoFileItrImpl::getLumisInRun(std::vector<LuminosityBlockNumber_t>& lumis) const {
    assert(shouldProcessRun());
    lumis.clear();

    if (type_ == kEnd)
      return;

    LuminosityBlockNumber_t previousLumi = invalidLumi;

    for (int i = 1; (i + indexToRun_) < indexedSize(); ++i) {
      int index = i + indexToRun_;
      EntryType entryType = getRunOrLumiEntryType(index);

      if (entryType == kRun) {
        if (isSameRun(indexToRun_, index)) {
          continue;
        } else {
          break;
        }
      } else {
        LuminosityBlockNumber_t luminosityBlock = lumi(index);
        if (luminosityBlock != invalidLumi && luminosityBlock != previousLumi) {
          lumis.push_back(luminosityBlock);
          previousLumi = luminosityBlock;
        }
      }
    }
    std::sort(lumis.begin(), lumis.end());
    lumis.erase(std::unique(lumis.begin(), lumis.end()), lumis.end());
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
                                                                long long nEvents)
      : IndexIntoFileItrImpl(
            indexIntoFile, entryType, indexToRun, indexToLumi, indexToEventRange, indexToEvent, nEvents) {}

  IndexIntoFile::IndexIntoFileItrImpl* IndexIntoFile::IndexIntoFileItrNoSort::clone() const {
    return new IndexIntoFileItrNoSort(*this);
  }

  int IndexIntoFile::IndexIntoFileItrNoSort::processHistoryIDIndex() const {
    if (type() == kEnd)
      return invalidIndex;
    return indexIntoFile()->runOrLumiEntries()[indexToRun()].processHistoryIDIndex();
  }

  RunNumber_t IndexIntoFile::IndexIntoFileItrNoSort::run() const {
    if (type() == kEnd)
      return invalidRun;
    return indexIntoFile()->runOrLumiEntries()[indexToRun()].run();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrNoSort::lumi() const {
    if (type() == kEnd || type() == kRun)
      return invalidLumi;
    return indexIntoFile()->runOrLumiEntries()[indexToLumi()].lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrNoSort::entry() const {
    if (type() == kEnd)
      return invalidEntry;
    if (type() == kRun)
      return indexIntoFile()->runOrLumiEntries()[indexToRun()].entry();
    if (type() == kLumi)
      return indexIntoFile()->runOrLumiEntries()[indexToLumi()].entry();
    return indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents() + indexToEvent();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrNoSort::peekAheadAtLumi() const {
    if (indexToLumi() == invalidIndex)
      return invalidLumi;
    return indexIntoFile()->runOrLumiEntries()[indexToLumi()].lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrNoSort::peekAheadAtEventEntry() const {
    if (indexToLumi() == invalidIndex)
      return invalidEntry;
    if (indexToEvent() >= nEvents())
      return invalidEntry;
    return indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents() + indexToEvent();
  }

  void IndexIntoFile::IndexIntoFileItrNoSort::initializeLumi_() {
    assert(indexToLumi() != invalidIndex);

    setIndexToEventRange(invalidIndex);
    setIndexToEvent(0);
    setNEvents(0);

    for (int i = 0; indexToLumi() + i < size(); ++i) {
      if (indexIntoFile()->runOrLumiEntries()[indexToLumi() + i].isRun()) {
        break;
      } else if (indexIntoFile()->runOrLumiEntries()[indexToLumi() + i].lumi() ==
                 indexIntoFile()->runOrLumiEntries()[indexToLumi()].lumi()) {
        if (indexIntoFile()->runOrLumiEntries()[indexToLumi() + i].beginEvents() == invalidEntry) {
          continue;
        }
        setIndexToEventRange(indexToLumi() + i);
        setIndexToEvent(0);
        setNEvents(indexIntoFile()->runOrLumiEntries()[indexToEventRange()].endEvents() -
                   indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents());
        break;
      } else {
        break;
      }
    }
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::nextEventRange() {
    if (indexToEventRange() == invalidIndex)
      return false;

    // Look for the next event range, same lumi but different entry
    for (int i = 1; indexToEventRange() + i < size(); ++i) {
      if (indexIntoFile()->runOrLumiEntries()[indexToEventRange() + i].isRun()) {
        return false;  // hit next run
      } else if (indexIntoFile()->runOrLumiEntries()[indexToEventRange() + i].lumi() ==
                 indexIntoFile()->runOrLumiEntries()[indexToEventRange()].lumi()) {
        if (indexIntoFile()->runOrLumiEntries()[indexToEventRange() + i].beginEvents() == invalidEntry) {
          continue;  // same lumi but has no events, keep looking
        }
        setIndexToEventRange(indexToEventRange() + i);
        setIndexToEvent(0);
        setNEvents(indexIntoFile()->runOrLumiEntries()[indexToEventRange()].endEvents() -
                   indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents());
        return true;  // found more events in this lumi
      }
      return false;  // hit next lumi
    }
    return false;  // hit the end of the IndexIntoFile
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::previousEventRange() {
    if (indexToEventRange() == invalidIndex)
      return false;
    assert(indexToEventRange() < size());

    // Look backward for a previous event range with events, same lumi but different entry
    for (int i = 1; indexToEventRange() - i > 0; ++i) {
      int newRange = indexToEventRange() - i;
      if (indexIntoFile()->runOrLumiEntries()[newRange].isRun()) {
        return false;  // hit run
      } else if (isSameLumi(newRange, indexToEventRange())) {
        if (indexIntoFile()->runOrLumiEntries()[newRange].beginEvents() == invalidEntry) {
          continue;  // same lumi but has no events, keep looking
        }
        setIndexToEventRange(newRange);
        setNEvents(indexIntoFile()->runOrLumiEntries()[indexToEventRange()].endEvents() -
                   indexIntoFile()->runOrLumiEntries()[indexToEventRange()].beginEvents());
        setIndexToEvent(nEvents() - 1);
        return true;  // found previous event in this lumi
      }
      return false;  // hit previous lumi
    }
    return false;  // hit the beginning of the IndexIntoFile, 0th entry has to be a run
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
    if (indexToLumi() == invalidIndex)
      return false;
    for (int i = 1; indexToLumi() + i < size(); ++i) {
      int newLumi = indexToLumi() + i;
      if (indexIntoFile()->runOrLumiEntries()[newLumi].isRun()) {
        return false;  // hit next run
      } else if (indexIntoFile()->runOrLumiEntries()[newLumi].lumi() ==
                 indexIntoFile()->runOrLumiEntries()[indexToLumi()].lumi()) {
        continue;
      }
      setIndexToLumi(newLumi);
      initializeLumi();
      return true;  // hit next lumi
    }
    return false;  // hit the end of the IndexIntoFile
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::lumiIterationStartingIndex(int index) const {
    return indexIntoFile()->runOrLumiEntries()[index].entry() != invalidEntry;
  }

  IndexIntoFile::EntryType IndexIntoFile::IndexIntoFileItrNoSort::getRunOrLumiEntryType(int index) const {
    if (index < 0 || index >= size()) {
      return kEnd;
    } else if (indexIntoFile()->runOrLumiEntries()[index].isRun()) {
      return kRun;
    }
    return kLumi;
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::isSameLumi(int index1, int index2) const {
    if (index1 < 0 || index1 >= size() || index2 < 0 || index2 >= size()) {
      return false;
    }
    return indexIntoFile()->runOrLumiEntries()[index1].lumi() == indexIntoFile()->runOrLumiEntries()[index2].lumi();
  }

  bool IndexIntoFile::IndexIntoFileItrNoSort::isSameRun(int index1, int index2) const {
    if (index1 < 0 || index1 >= size() || index2 < 0 || index2 >= size()) {
      return false;
    }
    return indexIntoFile()->runOrLumiEntries()[index1].run() == indexIntoFile()->runOrLumiEntries()[index2].run() &&
           indexIntoFile()->runOrLumiEntries()[index1].processHistoryIDIndex() ==
               indexIntoFile()->runOrLumiEntries()[index2].processHistoryIDIndex();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrNoSort::lumi(int index) const {
    if (index < 0 || index >= size()) {
      return invalidLumi;
    }
    return indexIntoFile()->runOrLumiEntries()[index].lumi();
  }

  IndexIntoFile::IndexIntoFileItrSorted::IndexIntoFileItrSorted(IndexIntoFile const* indexIntoFile,
                                                                EntryType entryType,
                                                                int indexToRun,
                                                                int indexToLumi,
                                                                int indexToEventRange,
                                                                long long indexToEvent,
                                                                long long nEvents)
      : IndexIntoFileItrImpl(
            indexIntoFile, entryType, indexToRun, indexToLumi, indexToEventRange, indexToEvent, nEvents) {
    indexIntoFile->fillRunOrLumiIndexes();
  }

  IndexIntoFile::IndexIntoFileItrImpl* IndexIntoFile::IndexIntoFileItrSorted::clone() const {
    return new IndexIntoFileItrSorted(*this);
  }

  int IndexIntoFile::IndexIntoFileItrSorted::processHistoryIDIndex() const {
    if (type() == kEnd)
      return invalidIndex;
    return indexIntoFile()->runOrLumiIndexes()[indexToRun()].processHistoryIDIndex();
  }

  RunNumber_t IndexIntoFile::IndexIntoFileItrSorted::run() const {
    if (type() == kEnd)
      return invalidRun;
    return indexIntoFile()->runOrLumiIndexes()[indexToRun()].run();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrSorted::lumi() const {
    if (type() == kEnd || type() == kRun)
      return invalidLumi;
    return indexIntoFile()->runOrLumiIndexes()[indexToLumi()].lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrSorted::entry() const {
    if (type() == kEnd)
      return invalidEntry;
    if (type() == kRun) {
      int i = indexIntoFile()->runOrLumiIndexes()[indexToRun()].indexToGetEntry();
      return indexIntoFile()->runOrLumiEntries()[i].entry();
    }
    if (type() == kLumi) {
      int i = indexIntoFile()->runOrLumiIndexes()[indexToLumi()].indexToGetEntry();
      return indexIntoFile()->runOrLumiEntries()[i].entry();
    }
    long long eventNumberIndex =
        indexIntoFile()->runOrLumiIndexes()[indexToEventRange()].beginEventNumbers() + indexToEvent();
    indexIntoFile()->fillEventEntries();
    return indexIntoFile()->eventEntries().at(eventNumberIndex).entry();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrSorted::peekAheadAtLumi() const {
    if (indexToLumi() == invalidIndex)
      return invalidLumi;
    return indexIntoFile()->runOrLumiIndexes()[indexToLumi()].lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrSorted::peekAheadAtEventEntry() const {
    if (indexToLumi() == invalidIndex)
      return invalidEntry;
    if (indexToEvent() >= nEvents())
      return invalidEntry;
    long long eventNumberIndex =
        indexIntoFile()->runOrLumiIndexes()[indexToEventRange()].beginEventNumbers() + indexToEvent();
    indexIntoFile()->fillEventEntries();
    return indexIntoFile()->eventEntries().at(eventNumberIndex).entry();
  }

  void IndexIntoFile::IndexIntoFileItrSorted::initializeLumi_() {
    assert(indexToLumi() != invalidIndex);
    setIndexToEventRange(indexToLumi());
    setIndexToEvent(0);
    setNEvents(indexIntoFile()->runOrLumiIndexes()[indexToLumi()].endEventNumbers() -
               indexIntoFile()->runOrLumiIndexes()[indexToLumi()].beginEventNumbers());
    if (nEvents() == 0) {
      setIndexToEventRange(invalidIndex);
    }
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::nextEventRange() { return false; }

  bool IndexIntoFile::IndexIntoFileItrSorted::previousEventRange() { return false; }

  bool IndexIntoFile::IndexIntoFileItrSorted::setToLastEventInRange(int index) {
    long long nEventsInRange = indexIntoFile()->runOrLumiIndexes()[index].endEventNumbers() -
                               indexIntoFile()->runOrLumiIndexes()[index].beginEventNumbers();
    if (nEventsInRange == 0) {
      return false;
    }
    while (index > 0 && !indexIntoFile()->runOrLumiIndexes()[index - 1].isRun() && isSameLumi(index, index - 1)) {
      --index;
    }
    assert(nEventsInRange == indexIntoFile()->runOrLumiIndexes()[index].endEventNumbers() -
                                 indexIntoFile()->runOrLumiIndexes()[index].beginEventNumbers());

    setIndexToEventRange(index);
    setNEvents(nEventsInRange);
    assert(nEvents() > 0);
    setIndexToEvent(nEventsInRange - 1);
    return true;
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::skipLumiInRun() {
    if (indexToLumi() == invalidIndex)
      return false;
    for (int i = 1; indexToLumi() + i < size(); ++i) {
      int newLumi = indexToLumi() + i;
      if (indexIntoFile()->runOrLumiIndexes()[newLumi].isRun()) {
        return false;  // hit next run
      } else if (indexIntoFile()->runOrLumiIndexes()[newLumi].lumi() ==
                 indexIntoFile()->runOrLumiIndexes()[indexToLumi()].lumi()) {
        continue;
      }
      setIndexToLumi(newLumi);
      initializeLumi();
      return true;  // hit next lumi
    }
    return false;  // hit the end of the IndexIntoFile
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::lumiIterationStartingIndex(int index) const {
    return indexIntoFile()->runOrLumiEntries()[indexIntoFile()->runOrLumiIndexes()[index].indexToGetEntry()].entry() !=
           invalidEntry;
  }

  IndexIntoFile::EntryType IndexIntoFile::IndexIntoFileItrSorted::getRunOrLumiEntryType(int index) const {
    if (index < 0 || index >= size()) {
      return kEnd;
    } else if (indexIntoFile()->runOrLumiIndexes()[index].isRun()) {
      return kRun;
    }
    return kLumi;
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::isSameLumi(int index1, int index2) const {
    if (index1 < 0 || index1 >= size() || index2 < 0 || index2 >= size()) {
      return false;
    }
    return indexIntoFile()->runOrLumiIndexes()[index1].lumi() == indexIntoFile()->runOrLumiIndexes()[index2].lumi();
  }

  bool IndexIntoFile::IndexIntoFileItrSorted::isSameRun(int index1, int index2) const {
    if (index1 < 0 || index1 >= size() || index2 < 0 || index2 >= size()) {
      return false;
    }
    return indexIntoFile()->runOrLumiIndexes()[index1].run() == indexIntoFile()->runOrLumiIndexes()[index2].run() &&
           indexIntoFile()->runOrLumiIndexes()[index1].processHistoryIDIndex() ==
               indexIntoFile()->runOrLumiIndexes()[index2].processHistoryIDIndex();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrSorted::lumi(int index) const {
    if (index < 0 || index >= size()) {
      return invalidLumi;
    }
    return indexIntoFile()->runOrLumiIndexes()[index].lumi();
  }

  // *************************************

  IndexIntoFile::IndexIntoFileItrEntryOrder::IndexIntoFileItrEntryOrder(IndexIntoFile const* iIndexIntoFile,
                                                                        EntryType entryType,
                                                                        int indexToRun,
                                                                        int indexToLumi,
                                                                        int indexToEventRange,
                                                                        long long indexToEvent,
                                                                        long long nEvents)
      : IndexIntoFileItrImpl(
            iIndexIntoFile, entryType, indexToRun, indexToLumi, indexToEventRange, indexToEvent, nEvents) {
    EntryOrderInitializationInfo info;
    info.resizeVectors(indexIntoFile()->runOrLumiEntries());
    reserveSpaceInVectors(indexIntoFile()->runOrLumiEntries().size());

    // fill firstIndexOfLumi, firstIndexOfRun, runsWithNoEvents
    info.gatherNeededInfo(indexIntoFile()->runOrLumiEntries());

    info.fillIndexesSortedByEventEntry(indexIntoFile()->runOrLumiEntries());
    info.fillIndexesToLastContiguousEvents(indexIntoFile()->runOrLumiEntries());

    EntryNumber_t currentRun = invalidEntry;

    // The main iterator created here (iEventSequence) is incremented
    // in the function handleToEndOfContiguousEventsInRun and
    // the functions it calls. The iterator is stored in "info",
    // which also holds other information related to the iteration.
    // The information is passed to these functions inside the "info"
    // object.
    for (info.iEventSequence_ = info.indexesSortedByEventEntry_.cbegin(),
        info.iEventSequenceEnd_ = info.indexesSortedByEventEntry_.cend();
         info.iEventSequence_ < info.iEventSequenceEnd_;) {
      info.eventSequenceIndex_ = info.iEventSequence_->runOrLumiIndex_;
      info.eventSequenceRunOrLumiEntry_ = &indexIntoFile()->runOrLumiEntries()[info.eventSequenceIndex_];

      assert(info.eventSequenceRunOrLumiEntry_->orderPHIDRun() != currentRun);
      currentRun = info.eventSequenceRunOrLumiEntry_->orderPHIDRun();

      // Handles the set of events contiguous in the Events TTree from
      // a single run and all the entries (Run or Lumi) associated with
      // those events and possibly some runs with no events that precede
      // the run in the runs TTree.
      handleToEndOfContiguousEventsInRun(info, currentRun);
    }
    // This takes care of only Runs with no Events at the end of
    // the Runs TTree that were not already added.
    addRunsWithNoEvents(info);
    indexedSize_ = fileOrderRunOrLumiEntry_.size();
  }

  IndexIntoFile::IndexIntoFileItrImpl* IndexIntoFile::IndexIntoFileItrEntryOrder::clone() const {
    return new IndexIntoFileItrEntryOrder(*this);
  }

  int IndexIntoFile::IndexIntoFileItrEntryOrder::processHistoryIDIndex() const {
    if (type() == kEnd)
      return invalidIndex;
    return runOrLumisEntry(indexToRun()).processHistoryIDIndex();
  }

  RunNumber_t IndexIntoFile::IndexIntoFileItrEntryOrder::run() const {
    if (type() == kEnd)
      return invalidRun;
    return runOrLumisEntry(indexToRun()).run();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrEntryOrder::lumi() const {
    if (type() == kEnd || type() == kRun)
      return invalidLumi;
    return runOrLumisEntry(indexToLumi()).lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrEntryOrder::entry() const {
    if (type() == kEnd)
      return invalidEntry;
    if (type() == kRun)
      return runOrLumisEntry(indexToRun()).entry();
    if (type() == kLumi) {
      auto entry = runOrLumisEntry(indexToLumi()).entry();
      if (entry == invalidEntry) {
        auto const& runLumiEntry = runOrLumisEntry(indexToLumi());
        for (int index = indexToLumi() + 1; index < indexedSize(); ++index) {
          auto const& laterRunOrLumiEntry = runOrLumisEntry(index);
          if (runLumiEntry.lumi() == laterRunOrLumiEntry.lumi() and runLumiEntry.run() == laterRunOrLumiEntry.run() and
              runLumiEntry.processHistoryIDIndex() == laterRunOrLumiEntry.processHistoryIDIndex() &&
              laterRunOrLumiEntry.entry() != invalidEntry) {
            return laterRunOrLumiEntry.entry();
          }
        }
        // We should always find one and never get here!
        throw Exception(errors::LogicError) << "In IndexIntoFile::IndexIntoFileItrEntryOrder::entry. Could not\n"
                                            << "find valid TTree entry number for lumi. This means the IndexIntoFile\n"
                                            << "product in the output file will be corrupted.\n"
                                            << "The output file will be unusable for most purposes.\n"
                                            << "If this occurs after an unrelated exception was thrown,\n"
                                            << "then ignore this exception and fix the primary exception.\n"
                                            << "This is an expected side effect.\n"
                                            << "Otherwise, please report this to the core framework developers\n";
      }
      return entry;
    }
    return runOrLumisEntry(indexToEventRange()).beginEvents() + indexToEvent();
  }

  bool IndexIntoFile::IndexIntoFileItrEntryOrder::shouldProcessLumi() const {
    assert(type() == kLumi);
    assert(indexToLumi() != invalidIndex);
    return shouldProcessRunOrLumi_[indexToLumi()];
  }

  bool IndexIntoFile::IndexIntoFileItrEntryOrder::shouldProcessRun() const {
    assert(type() == kRun);
    assert(indexToRun() != invalidIndex);
    return shouldProcessRunOrLumi_[indexToRun()];
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrEntryOrder::peekAheadAtLumi() const {
    if (indexToLumi() == invalidIndex)
      return invalidLumi;
    return runOrLumisEntry(indexToLumi()).lumi();
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrEntryOrder::peekAheadAtEventEntry() const {
    if (indexToLumi() == invalidIndex)
      return invalidEntry;
    if (indexToEvent() >= nEvents())
      return invalidEntry;
    return runOrLumisEntry(indexToEventRange()).beginEvents() + indexToEvent();
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::initializeLumi_() {
    assert(indexToLumi() != invalidIndex);

    setIndexToEventRange(invalidIndex);
    setIndexToEvent(0);
    setNEvents(0);

    for (int index = indexToLumi(); index < indexedSize(); ++index) {
      if (runOrLumisEntry(index).isRun()) {
        break;
      } else if (runOrLumisEntry(index).lumi() == runOrLumisEntry(indexToLumi()).lumi()) {
        if (runOrLumisEntry(index).beginEvents() == invalidEntry || !shouldProcessEvents(index)) {
          continue;
        }
        setIndexToEventRange(index);
        setIndexToEvent(0);
        setNEvents(runOrLumisEntry(indexToEventRange()).endEvents() -
                   runOrLumisEntry(indexToEventRange()).beginEvents());
        break;
      } else {
        break;
      }
    }
  }

  bool IndexIntoFile::IndexIntoFileItrEntryOrder::nextEventRange() {
    if (indexToEventRange() == invalidIndex)
      return false;

    // Look for the next event range, same lumi but different entry
    for (int index = indexToEventRange() + 1; index < indexedSize(); ++index) {
      if (runOrLumisEntry(index).isRun()) {
        return false;  // hit next run
      } else if (runOrLumisEntry(index).lumi() == runOrLumisEntry(indexToEventRange()).lumi()) {
        if (runOrLumisEntry(index).beginEvents() == invalidEntry || !shouldProcessEvents(index)) {
          continue;  // same lumi but has no events, keep looking
        }
        setIndexToEventRange(index);
        setIndexToEvent(0);
        setNEvents(runOrLumisEntry(indexToEventRange()).endEvents() -
                   runOrLumisEntry(indexToEventRange()).beginEvents());
        return true;  // found more events in this lumi
      }
      return false;  // hit next lumi
    }
    return false;  // hit the end of the IndexIntoFile
  }

  bool IndexIntoFile::IndexIntoFileItrEntryOrder::previousEventRange() {
    if (indexToEventRange() == invalidIndex)
      return false;
    assert(indexToEventRange() < indexedSize());

    // Look backward for a previous event range with events, same lumi but different entry
    for (int newRange = indexToEventRange() - 1; newRange > 0; --newRange) {
      if (runOrLumisEntry(newRange).isRun()) {
        return false;  // hit run
      } else if (isSameLumi(newRange, indexToEventRange())) {
        if (runOrLumisEntry(newRange).beginEvents() == invalidEntry || !shouldProcessEvents(newRange)) {
          continue;  // same lumi but has no events, keep looking
        }
        setIndexToEventRange(newRange);
        setNEvents(runOrLumisEntry(indexToEventRange()).endEvents() -
                   runOrLumisEntry(indexToEventRange()).beginEvents());
        setIndexToEvent(nEvents() - 1);
        return true;  // found previous event in this lumi
      }
      return false;  // hit previous lumi
    }
    return false;  // hit the beginning of the IndexIntoFile, 0th entry has to be a run
  }

  bool IndexIntoFile::IndexIntoFileItrEntryOrder::setToLastEventInRange(int index) {
    if (runOrLumisEntry(index).beginEvents() == invalidEntry || !shouldProcessEvents(index)) {
      return false;
    }
    setIndexToEventRange(index);
    setNEvents(runOrLumisEntry(indexToEventRange()).endEvents() - runOrLumisEntry(indexToEventRange()).beginEvents());
    assert(nEvents() > 0);
    setIndexToEvent(nEvents() - 1);
    return true;
  }

  bool IndexIntoFile::IndexIntoFileItrEntryOrder::skipLumiInRun() {
    if (indexToLumi() == invalidIndex)
      return false;
    for (int i = 1; indexToLumi() + i < indexedSize(); ++i) {
      int newLumi = indexToLumi() + i;
      if (runOrLumisEntry(newLumi).isRun()) {
        return false;  // hit next run
      } else if (runOrLumisEntry(newLumi).lumi() == runOrLumisEntry(indexToLumi()).lumi()) {
        continue;
      }
      setIndexToLumi(newLumi);
      initializeLumi();
      return true;  // hit next lumi
    }
    return false;  // hit the end of the IndexIntoFile
  }

  bool IndexIntoFile::IndexIntoFileItrEntryOrder::lumiIterationStartingIndex(int index) const {
    assert(index >= 0 && index < indexedSize());
    auto entry = runOrLumisEntry(index).entry();
    if (entry == invalidEntry) {
      // Usually the starting index is just the first one with a valid lumi TTree entry
      // number. If there aren't any that are valid, then use the last one.
      if (index + 1 < indexedSize()) {
        if (runOrLumisEntry(index).lumi() != runOrLumisEntry(index + 1).lumi()) {
          return true;
        }
      } else if (index + 1 == indexedSize()) {
        return true;
      }
    }
    return entry != invalidEntry;
  }

  IndexIntoFile::EntryType IndexIntoFile::IndexIntoFileItrEntryOrder::getRunOrLumiEntryType(int index) const {
    if (index < 0 || index >= indexedSize()) {
      return kEnd;
    } else if (runOrLumisEntry(index).isRun()) {
      return kRun;
    }
    return kLumi;
  }

  bool IndexIntoFile::IndexIntoFileItrEntryOrder::isSameLumi(int index1, int index2) const {
    if (index1 < 0 || index1 >= indexedSize() || index2 < 0 || index2 >= indexedSize()) {
      return false;
    }
    return runOrLumisEntry(index1).lumi() == runOrLumisEntry(index2).lumi();
  }

  bool IndexIntoFile::IndexIntoFileItrEntryOrder::isSameRun(int index1, int index2) const {
    if (index1 < 0 || index1 >= indexedSize() || index2 < 0 || index2 >= indexedSize()) {
      return false;
    }
    return runOrLumisEntry(index1).run() == runOrLumisEntry(index2).run() &&
           runOrLumisEntry(index1).processHistoryIDIndex() == runOrLumisEntry(index2).processHistoryIDIndex();
  }

  LuminosityBlockNumber_t IndexIntoFile::IndexIntoFileItrEntryOrder::lumi(int index) const {
    if (index < 0 || index >= indexedSize()) {
      return invalidLumi;
    }
    return runOrLumisEntry(index).lumi();
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::EntryOrderInitializationInfo::resizeVectors(
      std::vector<RunOrLumiEntry> const& runOrLumiEntries) {
    // The value in orderPHIDRun_ is unique to each run and corresponds
    // to a unique pair of values of run number and ProcessHistoryID.
    // It's incremented by one each time a new run is added to the
    // IndexIntoFile so that makes it convenient and efficient for
    // indexing elements in a vector with an element per run.
    // It is also convenient to use when comparing two runs
    // to see if they are the same run.
    // Analogous things are true for orderPHIDRunLumi_ except
    // that the lumi number is also used and it identifies lumis
    // instead of runs in IndexIntoFile.

    EntryNumber_t maxOrderPHIDRun = invalidEntry;
    EntryNumber_t maxOrderPHIDRunLumi = invalidEntry;
    unsigned int nSize = 0;

    for (auto const& runOrLumiEntry : runOrLumiEntries) {
      assert(runOrLumiEntry.orderPHIDRun() >= 0);
      if (runOrLumiEntry.orderPHIDRun() > maxOrderPHIDRun) {
        maxOrderPHIDRun = runOrLumiEntry.orderPHIDRun();
      }
      if (!runOrLumiEntry.isRun()) {
        assert(runOrLumiEntry.orderPHIDRunLumi() >= 0);
        if (runOrLumiEntry.orderPHIDRunLumi() > maxOrderPHIDRunLumi) {
          maxOrderPHIDRunLumi = runOrLumiEntry.orderPHIDRunLumi();
        }
      }
      if (runOrLumiEntry.beginEvents() != invalidEntry) {
        // Count entries with events
        ++nSize;
      }
    }
    firstIndexOfRun_.resize(maxOrderPHIDRun + 1, invalidIndex);
    firstIndexOfLumi_.resize(maxOrderPHIDRunLumi + 1, invalidIndex);
    startOfLastContiguousEventsInRun_.resize(maxOrderPHIDRun + 1, invalidIndex);
    startOfLastContiguousEventsInLumi_.resize(maxOrderPHIDRunLumi + 1, invalidIndex);
    indexesSortedByEventEntry_.reserve(nSize);
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::EntryOrderInitializationInfo::gatherNeededInfo(
      std::vector<RunOrLumiEntry> const& runOrLumiEntries) {
    int iEnd = static_cast<int>(runOrLumiEntries.size());

    EntryNumber_t previousLumi = invalidEntry;
    EntryNumber_t previousRun = invalidEntry;
    int index = 0;

    for (auto const& runOrLumiEntry : runOrLumiEntries) {
      // If first entry for a lumi
      if (!runOrLumiEntry.isRun() && runOrLumiEntry.orderPHIDRunLumi() != previousLumi) {
        previousLumi = runOrLumiEntry.orderPHIDRunLumi();

        // Fill map holding the first index into runOrLumiEntries for each lum
        firstIndexOfLumi_[runOrLumiEntry.orderPHIDRunLumi()] = index;
      }

      // If first entry for a run
      if (runOrLumiEntry.orderPHIDRun() != previousRun) {
        previousRun = runOrLumiEntry.orderPHIDRun();

        // Fill map holding the first index into runOrLumiEntries for each run
        firstIndexOfRun_[runOrLumiEntry.orderPHIDRun()] = index;

        // Look ahead to see if the run has events or not
        bool runHasEvents = false;
        for (int indexWithinRun = index + 1;
             indexWithinRun < iEnd && runOrLumiEntries[indexWithinRun].orderPHIDRun() == runOrLumiEntry.orderPHIDRun();
             ++indexWithinRun) {
          if (runOrLumiEntries[indexWithinRun].beginEvents() != invalidEntry) {
            runHasEvents = true;
            break;
          }
        }
        if (!runHasEvents) {
          runsWithNoEvents_.push_back({runOrLumiEntry.entry(), index});
        }
      }
      ++index;
    }

    std::sort(runsWithNoEvents_.begin(),
              runsWithNoEvents_.end(),
              [](TTreeEntryAndIndex const& left, TTreeEntryAndIndex const& right) -> bool {
                return left.ttreeEntry_ < right.ttreeEntry_;
              });

    nextRunWithNoEvents_ = runsWithNoEvents_.cbegin();
    endRunsWithNoEvents_ = runsWithNoEvents_.cend();
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::EntryOrderInitializationInfo::fillIndexesSortedByEventEntry(
      std::vector<RunOrLumiEntry> const& runOrLumiEntries) {
    int index = 0;
    for (auto const& runOrLumiEntry : runOrLumiEntries) {
      if (runOrLumiEntry.beginEvents() != invalidEntry) {
        indexesSortedByEventEntry_.push_back({runOrLumiEntry.beginEvents(), index});
      }
      ++index;
    }

    std::sort(indexesSortedByEventEntry_.begin(),
              indexesSortedByEventEntry_.end(),
              [](TTreeEntryAndIndex const& left, TTreeEntryAndIndex const& right) -> bool {
                return left.ttreeEntry_ < right.ttreeEntry_;
              });

    // The next "for loop" is just a sanity check, it should always pass.
    int previousIndex = invalidIndex;
    for (auto const& eventSequence : indexesSortedByEventEntry_) {
      int currentIndex = eventSequence.runOrLumiIndex_;
      if (previousIndex != invalidIndex) {
        assert(runOrLumiEntries[previousIndex].endEvents() == runOrLumiEntries[currentIndex].beginEvents());
      }
      previousIndex = currentIndex;
    }
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::EntryOrderInitializationInfo::fillIndexesToLastContiguousEvents(
      std::vector<RunOrLumiEntry> const& runOrLumiEntries) {
    EntryNumber_t previousRun = invalidEntry;
    EntryNumber_t previousLumi = invalidEntry;
    for (auto const& iter : indexesSortedByEventEntry_) {
      auto currentRun = runOrLumiEntries[iter.runOrLumiIndex_].orderPHIDRun();
      if (currentRun != previousRun) {
        startOfLastContiguousEventsInRun_[currentRun] = iter.runOrLumiIndex_;
        previousRun = currentRun;
      }
      auto currentLumi = runOrLumiEntries[iter.runOrLumiIndex_].orderPHIDRunLumi();
      if (currentLumi != previousLumi) {
        startOfLastContiguousEventsInLumi_[currentLumi] = iter.runOrLumiIndex_;
        previousLumi = currentLumi;
      }
    }
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::addRunsWithNoEvents(EntryOrderInitializationInfo& info,
                                                                      EntryNumber_t maxRunTTreeEntry) {
    auto const& runOrLumiEntries = indexIntoFile()->runOrLumiEntries();

    for (auto& nextRunWithNoEvents = info.nextRunWithNoEvents_;
         nextRunWithNoEvents != info.endRunsWithNoEvents_ &&
         (maxRunTTreeEntry == invalidEntry || nextRunWithNoEvents->ttreeEntry_ < maxRunTTreeEntry);
         ++nextRunWithNoEvents) {
      int index = nextRunWithNoEvents->runOrLumiIndex_;
      EntryNumber_t runToAdd = runOrLumiEntries[index].orderPHIDRun();
      for (int iEnd = static_cast<int>(runOrLumiEntries.size());
           index < iEnd && runOrLumiEntries[index].orderPHIDRun() == runToAdd;
           ++index) {
        // This will add in Run entries and the entries of Lumis in those Runs
        addToFileOrder(index, true, false);
      }
    }
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::fillLumisWithNoRemainingEvents(
      std::vector<TTreeEntryAndIndex>& lumisWithNoRemainingEvents,
      int startingIndex,
      EntryNumber_t currentRun,
      RunOrLumiEntry const* eventSequenceRunOrLumiEntry) const {
    auto const& runOrLumiEntries = indexIntoFile()->runOrLumiEntries();
    int iEnd = static_cast<int>(runOrLumiEntries.size());

    // start at the first entry after the Run entries
    // iterate over all the lumi entries in this Run
    // The outer loop iterates over lumis and inner loop iterates over entries in each lumi
    for (int indexOfLumiEntry = startingIndex;
         indexOfLumiEntry < iEnd && runOrLumiEntries[indexOfLumiEntry].orderPHIDRun() == currentRun;) {
      auto currentLumiIndex = indexOfLumiEntry;
      auto const& currentLumiEntry = runOrLumiEntries[currentLumiIndex];
      assert(!currentLumiEntry.isRun());
      auto currentLumi = currentLumiEntry.orderPHIDRunLumi();

      bool foundUnprocessedEvents = false;
      EntryNumber_t minLumiTTreeEntry = invalidEntry;
      // iterate over the lumi entries associated with a single lumi
      for (; indexOfLumiEntry < iEnd && runOrLumiEntries[indexOfLumiEntry].orderPHIDRunLumi() == currentLumi;
           ++indexOfLumiEntry) {
        if (runOrLumiEntries[indexOfLumiEntry].beginEvents() >= eventSequenceRunOrLumiEntry->beginEvents()) {
          foundUnprocessedEvents = true;
        }
        // Find the smallest valid Lumi TTree entry for this lumi
        auto lumiTTreeEntry = runOrLumiEntries[indexOfLumiEntry].entry();
        if (lumiTTreeEntry != invalidEntry &&
            (minLumiTTreeEntry == invalidEntry || lumiTTreeEntry < minLumiTTreeEntry)) {
          minLumiTTreeEntry = lumiTTreeEntry;
        }
      }
      // No event sequences left to process and at least one valid lumi TTree entry.
      if (!foundUnprocessedEvents && minLumiTTreeEntry != invalidEntry) {
        lumisWithNoRemainingEvents.push_back({minLumiTTreeEntry, currentLumiIndex});
      }
    }

    std::sort(lumisWithNoRemainingEvents.begin(),
              lumisWithNoRemainingEvents.end(),
              [](TTreeEntryAndIndex const& left, TTreeEntryAndIndex const& right) -> bool {
                return left.ttreeEntry_ < right.ttreeEntry_;
              });
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::reserveSpaceInVectors(
      std::vector<EntryNumber_t>::size_type sizeToReserve) {
    // Reserve some space. Most likely this is not big enough, but better than reserving nothing.
    fileOrderRunOrLumiEntry_.reserve(sizeToReserve);
    shouldProcessRunOrLumi_.reserve(sizeToReserve);
    shouldProcessEvents_.reserve(sizeToReserve);
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::addToFileOrder(int index, bool processRunOrLumi, bool processEvents) {
    fileOrderRunOrLumiEntry_.push_back(index);
    shouldProcessRunOrLumi_.push_back(processRunOrLumi);
    shouldProcessEvents_.push_back(processEvents);
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::handleToEndOfContiguousEventsInRun(EntryOrderInitializationInfo& info,
                                                                                     EntryNumber_t currentRun) {
    auto const& runOrLumiEntries = indexIntoFile()->runOrLumiEntries();
    int iEnd = static_cast<int>(runOrLumiEntries.size());

    int indexOfRunEntry = info.firstIndexOfRun_[currentRun];

    // Event entries are put in the exact same order as in the Events TTree.
    // We make some effort to make the Runs and Lumis come out in Run TTree
    // order and Lumi TTree order, but that is often not possible.

    // If it is the last contiguous sequence of events for the Run, also
    // add ALL entries corresponding to valid Run or Lumi TTree entries for
    // this Run. This is the place where the Run and Lumi products will get
    // processed and merged, ALL of them for this run whether or not they have
    // events in this particular subsequence of events. This forces all the Run
    // and Lumi product merging to occur the first time a file is read.
    if (info.startOfLastContiguousEventsInRun_[currentRun] == info.eventSequenceIndex_) {
      // Add runs with no events that have an earlier Run TTree entry number
      addRunsWithNoEvents(info, runOrLumiEntries[indexOfRunEntry].entry());

      // Add all valid run entries associated with the event sequence
      for (; indexOfRunEntry < iEnd && runOrLumiEntries[indexOfRunEntry].isRun(); ++indexOfRunEntry) {
        assert(runOrLumiEntries[indexOfRunEntry].orderPHIDRun() == currentRun);
        addToFileOrder(indexOfRunEntry, true, false);
      }

      // Add all lumi entries associated with this run
      handleToEndOfContiguousEventsInLumis(info, currentRun, indexOfRunEntry);

    } else {
      // Add only the first run entry and flag it to be not processed yet.
      addToFileOrder(indexOfRunEntry, false, false);

      // Add the minimum number of lumi entries so that the events they reference
      // will be processed in the correct order, lumis are not to be processed.
      // The lumis will be added again later to be processed.
      while (info.iEventSequence_ != info.iEventSequenceEnd_ &&
             info.eventSequenceRunOrLumiEntry_->orderPHIDRun() == currentRun) {
        addToFileOrder(info.eventSequenceIndex_, false, true);
        info.nextEventSequence(runOrLumiEntries);
      }
    }
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::handleToEndOfContiguousEventsInLumis(
      EntryOrderInitializationInfo& info, EntryNumber_t currentRun, int endOfRunEntries) {
    // Form a list of lumis that have no more events left to be processed and are in the current
    // run and have at least one valid Lumi TTree entry. Contains the index to the first
    // lumi entry and its TTree entry number, sorted by earliest lumi TTree entry number.
    std::vector<TTreeEntryAndIndex> lumisWithNoRemainingEvents;
    fillLumisWithNoRemainingEvents(
        lumisWithNoRemainingEvents, endOfRunEntries, currentRun, info.eventSequenceRunOrLumiEntry_);
    auto nextLumiWithNoEvents = lumisWithNoRemainingEvents.cbegin();
    auto endLumisWithNoEvents = lumisWithNoRemainingEvents.cend();

    // On each step of this iteration we process all the events in a contiguous sequence of events
    // from a single lumi (these are events that haven't already been processed and are contained
    // within the last contiguous sequence of events from the containing run).
    while (info.iEventSequence_ < info.iEventSequenceEnd_ &&
           info.eventSequenceRunOrLumiEntry_->orderPHIDRun() == currentRun) {
      auto currentLumi = info.eventSequenceRunOrLumiEntry_->orderPHIDRunLumi();

      // Last contiguous sequence of events in lumi
      if (info.startOfLastContiguousEventsInLumi_[currentLumi] == info.eventSequenceIndex_) {
        auto firstBeginEventsContiguousLumi = info.eventSequenceRunOrLumiEntry_->beginEvents();
        // Find the first Lumi TTree entry number for this Lumi
        EntryNumber_t lumiTTreeEntryNumber = lowestInLumi(info, currentLumi);

        // In addition, we want lumis before this in the lumi tree if they have no events
        // left to be processed
        handleLumisWithNoEvents(nextLumiWithNoEvents, endLumisWithNoEvents, lumiTTreeEntryNumber);

        // Handle the lumi with the next sequence of events to process
        handleLumiWithEvents(info, currentLumi, firstBeginEventsContiguousLumi);

      } else {
        // not last contiguous event sequence for lumi
        while (info.iEventSequence_ < info.iEventSequenceEnd_ &&
               info.eventSequenceRunOrLumiEntry_->orderPHIDRunLumi() == currentLumi) {
          addToFileOrder(info.eventSequenceIndex_, false, true);
          info.nextEventSequence(indexIntoFile()->runOrLumiEntries());
        }
      }
    }
    handleLumisWithNoEvents(nextLumiWithNoEvents, endLumisWithNoEvents, invalidEntry, true);
  }

  IndexIntoFile::EntryNumber_t IndexIntoFile::IndexIntoFileItrEntryOrder::lowestInLumi(
      EntryOrderInitializationInfo& info, int currentLumi) const {
    auto const& runOrLumiEntries = indexIntoFile()->runOrLumiEntries();
    int iEnd = static_cast<int>(runOrLumiEntries.size());

    for (int iLumiIndex = info.firstIndexOfLumi_[currentLumi];
         iLumiIndex < iEnd && runOrLumiEntries[iLumiIndex].orderPHIDRunLumi() == currentLumi;
         ++iLumiIndex) {
      EntryNumber_t lumiTTreeEntryNumber = runOrLumiEntries[iLumiIndex].entry();
      if (lumiTTreeEntryNumber != invalidEntry) {
        // First valid one is the lowest because of the sort order of the container
        return lumiTTreeEntryNumber;
      }
    }
    return invalidEntry;
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::handleLumisWithNoEvents(
      std::vector<TTreeEntryAndIndex>::const_iterator& nextLumiWithNoEvents,
      std::vector<TTreeEntryAndIndex>::const_iterator& endLumisWithNoEvents,
      EntryNumber_t lumiTTreeEntryNumber,
      bool completeAll) {
    auto const& runOrLumiEntries = indexIntoFile()->runOrLumiEntries();
    int iEnd = static_cast<int>(runOrLumiEntries.size());

    for (; nextLumiWithNoEvents < endLumisWithNoEvents &&
           (completeAll || nextLumiWithNoEvents->ttreeEntry_ < lumiTTreeEntryNumber);
         ++nextLumiWithNoEvents) {
      int iLumiIndex = nextLumiWithNoEvents->runOrLumiIndex_;
      auto orderPHIDRunLumi = runOrLumiEntries[iLumiIndex].orderPHIDRunLumi();
      for (; iLumiIndex < iEnd && runOrLumiEntries[iLumiIndex].orderPHIDRunLumi() == orderPHIDRunLumi; ++iLumiIndex) {
        if (runOrLumiEntries[iLumiIndex].entry() != invalidEntry) {
          addToFileOrder(iLumiIndex, true, false);
        }
      }
    }
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::handleLumiWithEvents(EntryOrderInitializationInfo& info,
                                                                       int currentLumi,
                                                                       EntryNumber_t firstBeginEventsContiguousLumi) {
    auto const& runOrLumiEntries = indexIntoFile()->runOrLumiEntries();
    int iLumiIndex = info.firstIndexOfLumi_[currentLumi];
    while (info.iEventSequence_ < info.iEventSequenceEnd_ &&
           info.eventSequenceRunOrLumiEntry_->orderPHIDRunLumi() == currentLumi) {
      // lumi entries for the currentLumi with no remaining Events to process and
      // with Lumi TTree entry numbers less than the Lumi TTree entry for the next
      // sequence of Events.
      handleLumiEntriesNoRemainingEvents(info, iLumiIndex, currentLumi, firstBeginEventsContiguousLumi);

      // Add entry with the next event sequence
      bool shouldProcessLumi = runOrLumiEntries[info.eventSequenceIndex_].entry() != invalidEntry;
      addToFileOrder(info.eventSequenceIndex_, shouldProcessLumi, true);
      info.nextEventSequence(runOrLumiEntries);
    }
    handleLumiEntriesNoRemainingEvents(info, iLumiIndex, currentLumi, firstBeginEventsContiguousLumi, true);
  }

  void IndexIntoFile::IndexIntoFileItrEntryOrder::handleLumiEntriesNoRemainingEvents(
      EntryOrderInitializationInfo& info,
      int& iLumiIndex,
      int currentLumi,
      EntryNumber_t firstBeginEventsContiguousLumi,
      bool completeAll) {
    auto const& runOrLumiEntries = indexIntoFile()->runOrLumiEntries();
    int iEnd = static_cast<int>(runOrLumiEntries.size());

    for (; iLumiIndex < iEnd && runOrLumiEntries[iLumiIndex].orderPHIDRunLumi() == currentLumi &&
           (completeAll || runOrLumiEntries[iLumiIndex].entry() < info.eventSequenceRunOrLumiEntry_->entry());
         ++iLumiIndex) {
      if (runOrLumiEntries[iLumiIndex].entry() == invalidEntry ||
          runOrLumiEntries[iLumiIndex].beginEvents() >= firstBeginEventsContiguousLumi) {
        continue;
      }
      addToFileOrder(iLumiIndex, true, false);
    }
  }

  //*************************************

  IndexIntoFile::IndexIntoFileItr::IndexIntoFileItr(IndexIntoFile const* indexIntoFile,
                                                    SortOrder sortOrder,
                                                    EntryType entryType,
                                                    int indexToRun,
                                                    int indexToLumi,
                                                    int indexToEventRange,
                                                    long long indexToEvent,
                                                    long long nEvents)
      : impl_() {
    if (sortOrder == numericalOrder) {
      value_ptr<IndexIntoFileItrImpl> temp(new IndexIntoFileItrSorted(
          indexIntoFile, entryType, indexToRun, indexToLumi, indexToEventRange, indexToEvent, nEvents));
      swap(temp, impl_);
    } else if (sortOrder == firstAppearanceOrder) {
      value_ptr<IndexIntoFileItrImpl> temp(new IndexIntoFileItrNoSort(
          indexIntoFile, entryType, indexToRun, indexToLumi, indexToEventRange, indexToEvent, nEvents));
      swap(temp, impl_);
    } else {
      value_ptr<IndexIntoFileItrImpl> temp(new IndexIntoFileItrEntryOrder(
          indexIntoFile, entryType, indexToRun, indexToLumi, indexToEventRange, indexToEvent, nEvents));
      swap(temp, impl_);
    }
  }

  void IndexIntoFile::IndexIntoFileItr::advanceToEvent() {
    for (EntryType entryType = getEntryType(); entryType != kEnd && entryType != kEvent; entryType = getEntryType()) {
      impl_->next();
    }
  }

  void IndexIntoFile::IndexIntoFileItr::advanceToLumi() {
    for (EntryType entryType = getEntryType(); entryType != kEnd && entryType != kLumi; entryType = getEntryType()) {
      impl_->next();
    }
  }

  void IndexIntoFile::IndexIntoFileItr::copyPosition(IndexIntoFileItr const& position) {
    impl_->copyPosition(*position.impl_);
  }

  bool Compare_Index_Run::operator()(IndexIntoFile::RunOrLumiIndexes const& lh,
                                     IndexIntoFile::RunOrLumiIndexes const& rh) {
    if (lh.processHistoryIDIndex() == rh.processHistoryIDIndex()) {
      return lh.run() < rh.run();
    }
    return lh.processHistoryIDIndex() < rh.processHistoryIDIndex();
  }

  bool Compare_Index::operator()(IndexIntoFile::RunOrLumiIndexes const& lh, IndexIntoFile::RunOrLumiIndexes const& rh) {
    return lh.processHistoryIDIndex() < rh.processHistoryIDIndex();
  }
}  // namespace edm
