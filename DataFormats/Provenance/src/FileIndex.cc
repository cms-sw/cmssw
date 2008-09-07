#include "DataFormats/Provenance/interface/FileIndex.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <algorithm>
#include <ostream>
#include <iomanip>

namespace edm {

  // The default value for sortState_ reflects the fact that
  // the index is always sorted using Run, Lumi, and Event
  // number by the PoolOutputModule before being written out.
  // In the other case when we create a new FileIndex, the
  // vector is empty, which is consistent with it having been
  // sorted.
  FileIndex::FileIndex() : entries_(), allEventsInEntryOrder_(false), resultCached_(false), sortState_(kSorted_Run_Lumi_Event) {}

  void
  FileIndex::addEntry(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, EntryNumber_t entry) {
    entries_.push_back(FileIndex::Element(run, lumi, event, entry));
    resultCached_ = false;
    sortState_ = kNotSorted;
  }

  void FileIndex::sortBy_Run_Lumi_Event() {
    stable_sort_all(entries_);
    resultCached_ = false;
    sortState_ = kSorted_Run_Lumi_Event;
  }

  void FileIndex::sortBy_Run_Lumi_EventEntry() {
    stable_sort_all(entries_, Compare_Run_Lumi_EventEntry());
    resultCached_ = false;
    sortState_ = kSorted_Run_Lumi_EventEntry;
  }

  bool FileIndex::allEventsInEntryOrder() const {
    if (!resultCached_) {
      resultCached_ = true;
      EntryNumber_t maxEntry = Element::invalidEntry;
      for (std::vector<FileIndex::Element>::const_iterator it = entries_.begin(), itEnd = entries_.end(); it != itEnd; ++it) {
        if (it->getEntryType() == kEvent) {
	  if (it->entry_ < maxEntry) {
	    allEventsInEntryOrder_ = false;
	    return allEventsInEntryOrder_;
          }
	  maxEntry = it->entry_;
        }
      }
      allEventsInEntryOrder_ = true;
    }
    return allEventsInEntryOrder_;
  }

  FileIndex::const_iterator
  FileIndex::findPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const {

    assert(sortState_ == kSorted_Run_Lumi_Event);

    Element el(run, lumi, event);
    const_iterator it = lower_bound_all(entries_, el);
    bool lumiMissing = (lumi == 0 && event != 0);
    if (lumiMissing) {
      const_iterator itEnd = entries_.end();
      while (it->event_ < event && it->run_ <= run && it != itEnd) ++it;
    }
    return it;
  }

  FileIndex::const_iterator
  FileIndex::findEventPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact) const {

    assert(sortState_ == kSorted_Run_Lumi_Event);

    const_iterator it = findPosition(run, lumi, event);
    const_iterator itEnd = entries_.end();
    while (it != itEnd && it->getEntryType() != FileIndex::kEvent) {
      ++it;
    }
    if (lumi == 0) lumi = it->lumi_;
    if (exact && (it->run_ != run || it->lumi_ != lumi || it->event_ != event)) it = entries_.end();
    return it;
  }

  FileIndex::const_iterator
  FileIndex::findLumiPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, bool exact) const {
    assert(sortState_ != kNotSorted);
    const_iterator it;
    if (sortState_ == kSorted_Run_Lumi_EventEntry) {
      Element el(run, lumi, 0U);
      it = lower_bound_all(entries_, el, Compare_Run_Lumi_EventEntry());
    }
    else {
      it = findPosition(run, lumi, 0U);
    }
    const_iterator itEnd = entries_.end();
    while (it != itEnd && it->getEntryType() != FileIndex::kLumi) {
      ++it;
    }
    if (exact && (it->run_ != run || it->lumi_ != lumi)) it = entries_.end();
    return it;
  }

  FileIndex::const_iterator
  FileIndex::findRunPosition(RunNumber_t run, bool exact) const {
    assert(sortState_ != kNotSorted);
    const_iterator it;
    if (sortState_ == kSorted_Run_Lumi_EventEntry) {
      Element el(run, 0U, 0U);
      it = lower_bound_all(entries_, el, Compare_Run_Lumi_EventEntry());
    }
    else {
      it = findPosition(run, 0U, 0U);
    }
    const_iterator itEnd = entries_.end();
    while (it != itEnd && it->getEntryType() != FileIndex::kRun) {
      ++it;
    }
    if (exact && (it->run_ != run)) it = entries_.end();
    return it;
  }

  FileIndex::const_iterator
  FileIndex::findLumiOrRunPosition(RunNumber_t run, LuminosityBlockNumber_t lumi) const {
    assert(sortState_ != kNotSorted);
    const_iterator it;
    if (sortState_ == kSorted_Run_Lumi_EventEntry) {
      Element el(run, lumi, 0U);
      it = lower_bound_all(entries_, el, Compare_Run_Lumi_EventEntry());
    }
    else {
      it = findPosition(run, lumi, 0U);
    }
    const_iterator itEnd = entries_.end();
    while (it != itEnd && it->getEntryType() != FileIndex::kLumi && it->getEntryType() != FileIndex::kRun) {
      ++it;
    }
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

  bool Compare_Run_Lumi_EventEntry::operator()(FileIndex::Element const& lh, FileIndex::Element const& rh)
  {
    if(lh.run_ == rh.run_) {
      if(lh.lumi_ == rh.lumi_) {
        if (lh.event_ == 0U && rh.event_ == 0U) return false;
        else if (lh.event_ == 0U) return true;
        else if (rh.event_ == 0U) return false;
	else return lh.entry_ < rh.entry_;
      }
      return lh.lumi_ < rh.lumi_;
    }
    return lh.run_ < rh.run_;
  }

  std::ostream&
  operator<< (std::ostream& os, FileIndex const& fileIndex) {

    os << "\nPrinting FileIndex contents.  This includes a list of all Runs, LuminosityBlocks\n"
       << "and Events stored in the root file.\n\n";
    os << std::setw(15) << "Run"
       << std::setw(15) << "Lumi"
       << std::setw(15) << "Event"
       << std::setw(15) << "TTree Entry"
       << "\n";
    for (std::vector<FileIndex::Element>::const_iterator it = fileIndex.begin(), itEnd = fileIndex.end(); it != itEnd; ++it) {
      if (it->getEntryType() == FileIndex::kEvent) {
        os << std::setw(15) << it->run_
           << std::setw(15) << it ->lumi_
           << std::setw(15) << it->event_
           << std::setw(15) << it->entry_
           << "\n";
      }
      else if (it->getEntryType() == FileIndex::kLumi) {
        os << std::setw(15) << it->run_
           << std::setw(15) << it ->lumi_
           << std::setw(15) << " "
           << std::setw(15) << it->entry_ << "  (LuminosityBlock)"
           << "\n";
      }
      else if (it->getEntryType() == FileIndex::kRun) {
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
