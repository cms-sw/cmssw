#ifndef DataFormats_Provenance_FileIndex_h
#define DataFormats_Provenance_FileIndex_h

/*----------------------------------------------------------------------

FileIndex.h 

----------------------------------------------------------------------*/

#include <vector>
#include <cassert>
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Transient.h"

#include "boost/shared_ptr.hpp"

#include <iosfwd>

namespace edm {

  class FileIndex {

    public:
      typedef long long EntryNumber_t;

      FileIndex();
      ~FileIndex() {}

      void addEntry(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, EntryNumber_t entry);

      enum EntryType {kRun, kLumi, kEvent, kEnd};

      class Element {
        public:
	  static EntryNumber_t const invalidEntry = -1LL;
          Element() : run_(0U), lumi_(0U), event_(0U), entry_(invalidEntry) {
	  }
          Element(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, long long entry) :
            run_(run), lumi_(lumi), 
          event_(event), entry_(entry) {
	    assert(lumi_ != 0U || event_ == 0U);
	  }
          Element(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) :
            run_(run), lumi_(lumi), event_(event), entry_(invalidEntry) {}
          EntryType getEntryType() const {
	    return lumi_ == 0U ? kRun : (event_ == 0U ? kLumi : kEvent);
          }
          RunNumber_t run_;
          LuminosityBlockNumber_t lumi_;
          EventNumber_t event_;
          EntryNumber_t entry_;
      };

      typedef std::vector<Element>::const_iterator const_iterator;

      typedef std::vector<Element>::iterator iterator;

      void sortBy_Run_Lumi_Event();
      void sortBy_Run_Lumi_EventEntry();

      const_iterator
      findPosition(RunNumber_t run, LuminosityBlockNumber_t lumi = 0U, EventNumber_t event = 0U) const;

      const_iterator
      findEventPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact) const;

      const_iterator
      findEventEntryPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, EntryNumber_t entry, bool exact) const;

      const_iterator
      findLumiPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, bool exact) const;

      const_iterator
      findRunPosition(RunNumber_t run, bool exact) const;

      const_iterator
      findLumiOrRunPosition(RunNumber_t run, LuminosityBlockNumber_t lumi) const;

      bool
      containsItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact) const {
	return event ? containsEvent(run, lumi, event, exact) : (lumi ? containsLumi(run, lumi, exact) : containsRun(run, exact));
      }

      bool
      containsEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact) const {
	return findEventPosition(run, lumi, event, exact) != entries_.end();
      }

      bool
      containsLumi(RunNumber_t run, LuminosityBlockNumber_t lumi, bool exact) const {
        return findLumiPosition(run, lumi, exact) != entries_.end();
      }

      bool
      containsRun(RunNumber_t run, bool exact) const {
        return findRunPosition(run, exact) != entries_.end();
      }

      const_iterator begin() const {return entries_.begin();}

      const_iterator end() const {return entries_.end();}

      iterator begin() {return entries_.begin();}

      iterator end() {return entries_.end();}

      iterator erase(iterator pos) {return entries_.erase(pos);}

      iterator erase(iterator begin, iterator end) {return entries_.erase(begin, end);}

      std::vector<Element>::size_type size() const {return entries_.size();}

      bool empty() const {return entries_.empty();}

      bool allEventsInEntryOrder() const;

      enum SortState {kNotSorted, kSorted_Run_Lumi_Event, kSorted_Run_Lumi_EventEntry};

      struct Transients {
	Transients();
	bool allInEntryOrder_;
	bool resultCached_;
	SortState sortState_;
      };

    private:

      bool& allInEntryOrder() const {return transients_.get().allInEntryOrder_;}
      bool& resultCached() const {return transients_.get().resultCached_;}
      SortState& sortState() const {return transients_.get().sortState_;}

      std::vector<Element> entries_;
      mutable Transient<Transients> transients_;
  };

  bool operator<(FileIndex::Element const& lh, FileIndex::Element const& rh);

  inline
  bool operator>(FileIndex::Element const& lh, FileIndex::Element const& rh) {return rh < lh;}

  inline
  bool operator>=(FileIndex::Element const& lh, FileIndex::Element const& rh) {return !(lh < rh);}

  inline
  bool operator<=(FileIndex::Element const& lh, FileIndex::Element const& rh) {return !(rh < lh);}

  inline
  bool operator==(FileIndex::Element const& lh, FileIndex::Element const& rh) {return !(lh < rh || rh < lh);}

  inline
  bool operator!=(FileIndex::Element const& lh, FileIndex::Element const& rh) {return lh < rh || rh < lh;}

  class Compare_Run_Lumi_EventEntry {
  public:
    bool operator()(FileIndex::Element const& lh, FileIndex::Element const& rh);
  };

  std::ostream&
  operator<<(std::ostream& os, FileIndex const& fileIndex);
}

#endif
