#include "DataFormats/Provenance/interface/FileIndex.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <algorithm>
#include <iomanip>
#include <ostream>

namespace edm {

  FileIndex::FileIndex() : entries_(), transient_() {}

  // The default value for sortState_ reflects the fact that
  // the index is always sorted using Run, Lumi, and Event
  // number by the PoolOutputModule before being written out.
  // In the other case when we create a new FileIndex, the
  // vector is empty, which is consistent with it having been
  // sorted.

  FileIndex::Transients::Transients() : allInEntryOrder_(false), resultCached_(false), sortState_(kSorted_Run_Lumi_Event) {}

  void
  FileIndex::Transients::reset() {
    allInEntryOrder_ = false;
    resultCached_ = false;
    sortState_ = kSorted_Run_Lumi_Event;
  }

  void
  FileIndex::addEntry(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, EntryNumber_t entry) {
    entries_.push_back(FileIndex::Element(run, lumi, event, entry));
    resultCached() = false;
    sortState() = kNotSorted;
  }

  void FileIndex::sortBy_Run_Lumi_Event() {
    stable_sort_all(entries_);
    resultCached() = false;
    sortState() = kSorted_Run_Lumi_Event;
  }

  void FileIndex::sortBy_Run_Lumi_EventEntry() {
    stable_sort_all(entries_, Compare_Run_Lumi_EventEntry());
    resultCached() = false;
    sortState() = kSorted_Run_Lumi_EventEntry;
  }

  bool FileIndex::allEventsInEntryOrder() const {
    if(!resultCached()) {
      resultCached() = true;
      EntryNumber_t maxEntry = Element::invalidEntry;
      for(std::vector<FileIndex::Element>::const_iterator it = entries_.begin(), itEnd = entries_.end(); it != itEnd; ++it) {
        if(it->getEntryType() == kEvent) {
          if(it->entry_ < maxEntry) {
            allInEntryOrder() = false;
            return false;
          }
          maxEntry = it->entry_;
        }
      }
      allInEntryOrder() = true;
      return true;
    }
    return allInEntryOrder();
  }

  FileIndex::const_iterator
  FileIndex::findPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {
    assert(sortState() != kNotSorted);

    const_iterator itEnd = entries_.end();
    const_iterator it;
    Element el(run, lumi, event);
    if (sortState() == kSorted_Run_Lumi_Event) {
      it =  lower_bound_all(entries_, el);
      bool lumiMissing = (lumi == 0 && event != 0);
      if(lumiMissing) {
        while(it != itEnd && it->run_ < run) {
          ++it;
        }
        while(it != itEnd && (it->run_ == run && it->event_ < event)) {
          ++it;
        }
      }
    } else {
      it = lower_bound_all(entries_, el, Compare_Run_Lumi_EventEntry());
    }
    return it;
  }

  FileIndex::const_iterator
  FileIndex::findEventPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {

    const_iterator it = findPosition(run, lumi, event);
    const_iterator itEnd = entries_.end();
    while(it != itEnd && it->getEntryType() != FileIndex::kEvent) {
      ++it;
    }
    if(it == itEnd) {
      return itEnd;
    }
    if(lumi == 0) {
      lumi = it->lumi_;
    }
    if(it->run_ != run || it->lumi_ != lumi || it->event_ != event) {
      if (sortState() == kSorted_Run_Lumi_Event) {
        return itEnd;
      }
      // not sorted by event, so we need to do a linear search
      while (it != itEnd && it->run_ == run && it->lumi_ == lumi && it->event_ != event) {
         ++it;
      }
      if(it->run_ != run || it->lumi_ != lumi || it->event_ != event) {
        return itEnd;
      }
    }
    return it;
  }

  FileIndex::const_iterator
  FileIndex::findLumiPosition(RunNumber_t run, LuminosityBlockNumber_t lumi) const {
    const_iterator it = findPosition(run, lumi, 0U);
    const_iterator itEnd = entries_.end();
    while(it != itEnd && it->getEntryType() != FileIndex::kLumi) {
      ++it;
    }
    if(it == itEnd) {
      return itEnd;
    }
    if(it->run_ != run || it->lumi_ != lumi) {
      return itEnd;
    }
    return it;
  }

  FileIndex::const_iterator
  FileIndex::findRunPosition(RunNumber_t run) const {
    const_iterator it = findPosition(run, 0U, 0U);
    const_iterator itEnd = entries_.end();
    while(it != itEnd && it->getEntryType() != FileIndex::kRun) {
      ++it;
    }
    if(it == itEnd) {
      return itEnd;
    }
    if(it->run_ != run) {
      return itEnd;
    }
    return it;
  }

  FileIndex::const_iterator
  FileIndex::findLumiOrRunPosition(RunNumber_t run, LuminosityBlockNumber_t lumi) const {
    const_iterator it = findPosition(run, lumi, 0U);
    const_iterator itEnd = entries_.end();
    while(it != itEnd && it->getEntryType() != FileIndex::kLumi && it->getEntryType() != FileIndex::kRun) {
      ++it;
    }
    return it;
  }

  FileIndex::const_iterator
  FileIndex::findEventEntryPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, EntryNumber_t entry) const {
    assert(sortState() != kNotSorted);
    const_iterator it;
    const_iterator itEnd = entries_.end();
    if(sortState() == kSorted_Run_Lumi_EventEntry) {
      assert(lumi != 0U);
      Element el(run, lumi, event, entry);
      it = lower_bound_all(entries_, el, Compare_Run_Lumi_EventEntry());
    } else {
      it = findEventPosition(run, lumi, event);
      while(it != itEnd && it->entry_ != entry && it->event_ == event) {
        ++it;
      }
    }
    if(it == itEnd) return itEnd;
    if(lumi == 0) lumi = it->lumi_;
    if(it->run_ != run || it->lumi_ != lumi || it->event_ != event || it->entry_ != entry) return itEnd;
    return it;
  }

  bool operator<(FileIndex::Element const& lh, FileIndex::Element const& rh) {
    if(lh.run_ == rh.run_) {
      if(lh.lumi_ == rh.lumi_) {
        return lh.event_ < rh.event_;
      }
      return lh.lumi_ < rh.lumi_;
    }
    return lh.run_ < rh.run_;
  }

  bool Compare_Run_Lumi_EventEntry::operator()(FileIndex::Element const& lh, FileIndex::Element const& rh) {
    if(lh.run_ == rh.run_) {
      if(lh.lumi_ == rh.lumi_) {
        if(lh.event_ == 0U && rh.event_ == 0U) {
          return false;
        } else if(lh.event_ == 0U) {
          return true;
        } else if(rh.event_ == 0U) {
          return false;
        } else {
          return lh.entry_ < rh.entry_;
        }
      }
      return lh.lumi_ < rh.lumi_;
    }
    return lh.run_ < rh.run_;
  }

  std::ostream&
  operator<<(std::ostream& os, FileIndex const& fileIndex) {

    os << "\nPrinting FileIndex contents.  This includes a list of all Runs, LuminosityBlocks\n"
       << "and Events stored in the root file.\n\n";
    os << std::setw(15) << "Run"
       << std::setw(15) << "Lumi"
       << std::setw(15) << "Event"
       << std::setw(15) << "TTree Entry"
       << "\n";
    for(std::vector<FileIndex::Element>::const_iterator it = fileIndex.begin(), itEnd = fileIndex.end(); it != itEnd; ++it) {
      if(it->getEntryType() == FileIndex::kEvent) {
        os << std::setw(15) << it->run_
           << std::setw(15) << it ->lumi_
           << std::setw(15) << it->event_
           << std::setw(15) << it->entry_
           << "\n";
      }
      else if(it->getEntryType() == FileIndex::kLumi) {
        os << std::setw(15) << it->run_
           << std::setw(15) << it ->lumi_
           << std::setw(15) << " "
           << std::setw(15) << it->entry_ << "  (LuminosityBlock)"
           << "\n";
      }
      else if(it->getEntryType() == FileIndex::kRun) {
        os << std::setw(15) << it->run_
           << std::setw(15) << " "
           << std::setw(15) << " "
           << std::setw(15) << it->entry_ << "  (Run)"
           << "\n";
      }
    }
    return os;
  }
}
