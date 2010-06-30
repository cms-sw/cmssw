#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <algorithm>
#include <ostream>
#include <iomanip>
#include <map>

namespace edm {

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

    resultCached() = false;

    assert((currentRun() == run && currentIndex() == index) || currentRun() == invalidRun);
    if (lumi == invalidLumi) {
      assert(currentLumi() == invalidLumi);
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
      while (beginOfLumi != iEnd && beginOfLumi->lumi() == invalidLumi) {
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
      assert(firstRunEntry != runToFirstEntry().end());

      iter->setOrderPHIDRun(firstRunEntry->second);
    }
    stable_sort_all(runOrLumiEntries_);
  }

  void IndexIntoFile::sortEvents() {
    fillRunOrLumiIndexes();
    std::vector<RunOrLumiIndexes>::iterator beginOfLumi = runOrLumiIndexes().begin();
    std::vector<RunOrLumiIndexes>::iterator endOfLumi = beginOfLumi;
    std::vector<RunOrLumiIndexes>::iterator iEnd = runOrLumiIndexes().end();
    while (true) {
      while (beginOfLumi != iEnd && beginOfLumi->lumi() == invalidLumi) {
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
      std::sort(eventNumbers().begin() + beginOfLumi->beginEventNumbers(),
                eventNumbers().begin() + beginOfLumi->endEventNumbers());
      beginOfLumi = endOfLumi;
    }
  }

  void IndexIntoFile::sortEventEntries() {
    fillRunOrLumiIndexes();
    std::vector<RunOrLumiIndexes>::iterator beginOfLumi = runOrLumiIndexes().begin();
    std::vector<RunOrLumiIndexes>::iterator endOfLumi = beginOfLumi;
    std::vector<RunOrLumiIndexes>::iterator iEnd = runOrLumiIndexes().end();
    while (true) {
      while (beginOfLumi != iEnd && beginOfLumi->lumi() == invalidLumi) {
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
    if(!resultCached()) {
      resultCached() = true;
      EntryNumber_t maxEntry = invalidEntry;
      for(IndexIntoFileItr it = begin(sortOrder), itEnd = end(sortOrder); it != itEnd; ++it) {
        if(it.getEntryType() == kEvent) {
	  if(it.entry() < maxEntry) {
	    allInEntryOrder() = false;
	    return false;
          }
	  maxEntry = it.entry();
        }
      }
      allInEntryOrder() = true;
      return true;
    }
    return allInEntryOrder();
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
        if (!eventNumbers().empty()) {
          std::vector<EventNumber_t>::const_iterator eventIter = std::lower_bound(eventNumbers().begin() + beginEventNumbers,
                                                                                  eventNumbers().begin() + endEventNumbers,
                                                                                  event);
          if (eventIter == (eventNumbers().begin() + endEventNumbers) ||
              *eventIter != event) continue;

          indexToEvent = eventIter - eventNumbers().begin() - beginEventNumbers;
        }
        else {
          assert(!eventEntries().empty());
          std::vector<EventEntry>::const_iterator eventIter = std::lower_bound(eventEntries().begin() + beginEventNumbers,
                                                                               eventEntries().begin() + endEventNumbers,
                                                                               EventEntry(event, invalidEntry));
          if (eventIter == (eventEntries().begin() + endEventNumbers) ||
              eventIter->event() != event) continue;

          indexToEvent = eventIter - eventEntries().begin() - beginEventNumbers;
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
          if (!eventNumbers().empty()) {
            std::vector<EventNumber_t>::const_iterator eventIter = std::lower_bound(eventNumbers().begin() + beginEventNumbers,
                                                                                    eventNumbers().begin() + endEventNumbers,
                                                                                    event);
            if (eventIter == (eventNumbers().begin() + endEventNumbers) ||
                *eventIter != event) continue;
            indexToEvent = eventIter - eventNumbers().begin() - beginEventNumbers;
          }
          else {
            assert(!eventEntries().empty());
            std::vector<EventEntry>::const_iterator eventIter = std::lower_bound(eventEntries().begin() + beginEventNumbers,
                                                                                 eventEntries().begin() + endEventNumbers,
                                                                                 EventEntry(event, invalidEntry));
            if (eventIter == (eventEntries().begin() + endEventNumbers) ||
                eventIter->event() != event) continue;
            indexToEvent = eventIter - eventEntries().begin() - beginEventNumbers;
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

    SortedRunOrLumiItr iter1 = beginRunOrLumi();
    SortedRunOrLumiItr iEnd1 = endRunOrLumi();

    SortedRunOrLumiItr iter2 = indexIntoFile.beginRunOrLumi();
    SortedRunOrLumiItr iEnd2 = indexIntoFile.endRunOrLumi();

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
          // Found a matching lumi, now look for matching events

          long long beginEventNumbers1 = indexes1.beginEventNumbers();
          long long endEventNumbers1 = indexes1.endEventNumbers();

          long long beginEventNumbers2 = indexes2.beginEventNumbers();
          long long endEventNumbers2 = indexes2.endEventNumbers();

          // there must be at least 1 event in each lumi for there to be any matches
          if (beginEventNumbers1 >= endEventNumbers1) continue;
          if (beginEventNumbers2 >= endEventNumbers2) continue;

          if (!eventNumbers().empty()) {
            assert(!indexIntoFile.eventNumbers().empty());
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
          else {
            assert(!eventEntries().empty());
            assert(!indexIntoFile.eventEntries().empty());
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

          }
          previousIndexes = &indexes1;
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

      long long beginEventNumbers = indexes.beginEventNumbers();
      long long endEventNumbers = indexes.endEventNumbers();

      // there must be more than 1 event in the lumi for there to be any duplicates
      if (beginEventNumbers + 1 >= endEventNumbers) continue;

      if (!eventNumbers().empty()) {
        if (std::adjacent_find(eventNumbers().begin() + beginEventNumbers,
                               eventNumbers().begin() + endEventNumbers) != eventNumbers().end()) {
           return true;
        }
      }
      else {
        assert(!eventEntries().empty());
        if (std::adjacent_find(eventEntries().begin() + beginEventNumbers,
                               eventEntries().begin() + endEventNumbers) != eventEntries().end()) {
          return true;
        }
      }
      previousIndexes = &indexes;
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


  IndexIntoFile::IndexIntoFileItrImpl::IndexIntoFileItrImpl(IndexIntoFile const* indexIntoFile) :
    indexIntoFile_(indexIntoFile),
    size_(static_cast<int>(indexIntoFile_->runOrLumiEntries_.size())),
    type_(kEnd),
    indexToRun_(invalidIndex),
    indexToLumi_(invalidIndex),
    indexToEventRange_(invalidIndex),
    indexToEvent_(0),
    nEvents_(0) {

    if (size_ == 0) {
      return;
    }
    type_ = kRun;
    assert(indexIntoFile_->runOrLumiEntries_[0].isRun());
    initializeRun();
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

  void IndexIntoFile::IndexIntoFileItrImpl::next () {

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

      bool hasEvents = lumiHasEvents();

      if (indexToLumi_ + 1 == size_) {
        if (hasEvents) {
          type_ = kEvent;
        }
        else {
          type_ = kEnd;
          setInvalid();
        }
      }
      else {

        EntryType nextType = getRunOrLumiEntryType(indexToLumi_ + 1);
        bool sameLumi = isSameLumi(indexToLumi_, indexToLumi_ + 1);

        if (sameLumi && nextType != kRun) {
          ++indexToLumi_;
        }
        else if (hasEvents) {
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
      else {
        type_ = kLumi;
      }
    }
  }

  void IndexIntoFile::IndexIntoFileItrImpl::skipEventForward(int & phIndexOfSkippedEvent,
                                                             RunNumber_t & runOfSkippedEvent,
                                                             LuminosityBlockNumber_t & lumiOfSkippedEvent,
                                                             EntryNumber_t & skippedEventEntry) {
    phIndexOfSkippedEvent = invalidIndex;
    runOfSkippedEvent = invalidRun;
    lumiOfSkippedEvent = invalidLumi;
    skippedEventEntry = invalidEntry;

    if (indexToEvent_  < nEvents_) {
      phIndexOfSkippedEvent = processHistoryIDIndex();
      runOfSkippedEvent = run();
      lumiOfSkippedEvent = peekAheadAtLumi();
      skippedEventEntry = peekAheadAtEventEntry();

      if ((indexToEvent_ + 1)  < nEvents_) {
        ++indexToEvent_;
        return;
      }

      if (nextEventRange()) {
        return;
      }
      else if (type_ == kRun) {
        if (skipLumiInRun()) {
          return;
        }
      }
    }
    if (type_ == kRun) {
      while (skipLumiInRun()) {
        if (indexToEvent_  < nEvents_) {
          skipEventForward(phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, skippedEventEntry);
          return;
        }
      }
    }
    while (type_ != kEvent && type_ != kEnd) {
      next();
    }
    if (type_ == kEnd) {
      return;
    }
    phIndexOfSkippedEvent = processHistoryIDIndex();
    runOfSkippedEvent = run();
    lumiOfSkippedEvent = lumi();
    skippedEventEntry = entry();
    next();
    return;
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
    type_ = kEnd;
    setInvalid();
  }

  void IndexIntoFile::IndexIntoFileItrImpl::advanceToNextLumiOrRun() {
    if (type_ == kEnd) return;
    int startSearch = indexToLumi_;
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
      if (indexToLumi_ != invalidIndex && isSameLumi(indexToLumi_, startSearch + i)) {
        continue;
      }
      type_ = kLumi;
      indexToLumi_ = startSearch + i;
      initializeLumi();
    }
    type_ = kEnd;
    setInvalid();
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

  void IndexIntoFile::IndexIntoFileItrImpl::setInvalid() {
    indexToRun_ = invalidIndex;
    indexToLumi_ = invalidIndex;
    indexToEventRange_ = invalidIndex;
    indexToEvent_ = 0;
    nEvents_ = 0;
  }

  IndexIntoFile::IndexIntoFileItrNoSort::IndexIntoFileItrNoSort(IndexIntoFile const* indexIntoFile) :
    IndexIntoFileItrImpl(indexIntoFile)
  {
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
  bool IndexIntoFile::IndexIntoFileItrNoSort::skipLumiInRun() { 
    for(int i = 1; indexToEventRange() + i < size(); ++i) {
      if (indexIntoFile()->runOrLumiEntries()[indexToEventRange() + i ].isRun()) {
        return false;  // hit next run
      }
      else if (indexIntoFile()->runOrLumiEntries()[indexToEventRange() + i].lumi() ==
               indexIntoFile()->runOrLumiEntries()[indexToEventRange()].lumi()) {
        continue;
      }
      setIndexToLumi(indexToEventRange() + i);
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

  bool IndexIntoFile::IndexIntoFileItrNoSort::lumiHasEvents() const {
    return indexIntoFile()->runOrLumiEntries()[indexToLumi()].beginEvents() != invalidEntry;
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

  IndexIntoFile::IndexIntoFileItrSorted::IndexIntoFileItrSorted(IndexIntoFile const* indexIntoFile) :
    IndexIntoFileItrImpl(indexIntoFile) {
    indexIntoFile->fillRunOrLumiIndexes();
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
    return indexIntoFile()->eventEntries().at(eventNumberIndex).entry();
  }

  void IndexIntoFile::IndexIntoFileItrSorted::initializeLumi_() {
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

  bool IndexIntoFile::IndexIntoFileItrSorted::skipLumiInRun() { 
    for(int i = 1; indexToEventRange() + i < size(); ++i) {
      if (indexIntoFile()->runOrLumiIndexes()[indexToEventRange() + i ].isRun()) {
        return false;  // hit next run
      }
      else if (indexIntoFile()->runOrLumiIndexes()[indexToEventRange() + i].lumi() ==
               indexIntoFile()->runOrLumiIndexes()[indexToEventRange()].lumi()) {
        continue;
      }
      setIndexToLumi(indexToEventRange() + i);
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

  bool IndexIntoFile::IndexIntoFileItrSorted::lumiHasEvents() const {
    return indexIntoFile()->runOrLumiIndexes()[indexToLumi()].beginEventNumbers() !=
           indexIntoFile()->runOrLumiIndexes()[indexToLumi()].endEventNumbers();
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

  IndexIntoFile::IndexIntoFileItr::IndexIntoFileItr(IndexIntoFile const* indexIntoFile, SortOrder sortOrder) :
    impl_() {
    if (sortOrder == numericalOrder) {
      value_ptr<IndexIntoFileItrImpl> temp(new IndexIntoFileItrSorted(indexIntoFile));
      swap(temp, impl_);
    }
    else {
      value_ptr<IndexIntoFileItrImpl> temp(new IndexIntoFileItrNoSort(indexIntoFile));
      swap(temp, impl_);
    }
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

  IndexIntoFile::Transients::Transients() : allInEntryOrder_(false),
                                            resultCached_(false),
                                            previousAddedIndex_(invalidIndex),
                                            runToFirstEntry_(),
                                            lumiToFirstEntry_(),
                                            beginEvents_(invalidEntry),
                                            endEvents_(invalidEntry),
                                            currentIndex_(invalidIndex),
                                            currentRun_(invalidRun),
                                            currentLumi_(invalidLumi),
                                            runOrLumiIndexes_(),
                                            eventNumbers_(),
                                            eventEntries_() {
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

  std::ostream&
  operator<<(std::ostream& os, IndexIntoFile const& fileIndex) {
    // Need to fix this, used by edmFileUtil
    /*
    os << "\nPrinting IndexIntoFile contents.  This includes a list of all Runs, LuminosityBlocks\n"
       << "and Events stored in the root file.\n\n";
    os << std::setw(15) << "Process History"
       << std::setw(15) << "Run"
       << std::setw(15) << "Lumi"
       << std::setw(15) << "Event"
       << std::setw(19) << "TTree Entry"
       << "\n";
    for(std::vector<IndexIntoFile::Element>::const_iterator it = fileIndex.begin(), itEnd = fileIndex.end(); it != itEnd; ++it) {
      if(it->getEntryType() == IndexIntoFile::kEvent) {
        os << std::setw(15) << it->processHistoryIDIndex()
           << std::setw(15) << it->run()
           << std::setw(15) << it ->lumi()
           << std::setw(15) << it->event()
           << std::setw(19) << it->entry()
           << "\n";
      }
      else if(it->getEntryType() == IndexIntoFile::kLumi) {
        os << std::setw(15) << it->processHistoryIDIndex()
           << std::setw(15) << it->run()
           << std::setw(15) << it ->lumi()
           << std::setw(15) << " "
           << std::setw(19) << it->entry() << "  (LuminosityBlock)"
           << "\n";
      }
      else if(it->getEntryType() == IndexIntoFile::kRun) {
        os << std::setw(15) << it->processHistoryIDIndex()
           << std::setw(15) << it->run()
           << std::setw(15) << " "
           << std::setw(15) << " "
           << std::setw(19) << it->entry() << "  (Run)"
           << "\n";
      }
    }
    os << "\nProcess History IDs (the first value on each line above is an index into this list of IDs)\n";
    int i = 0;
    for (std::vector<ProcessHistoryID>::const_iterator iter = fileIndex.processHistoryIDs().begin(),
                                                       iEnd = fileIndex.processHistoryIDs().end();
         iter != iEnd;
         ++iter, ++i) {
      os << "  " << i << "  " << *iter << "\n";
    }
    */
    return os;
  }
}
