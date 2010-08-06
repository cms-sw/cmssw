#ifndef DataFormats_Provenance_IndexIntoFile_h
#define DataFormats_Provenance_IndexIntoFile_h

/*----------------------------------------------------------------------

IndexIntoFile.h

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Transient.h"
#include "FWCore/Utilities/interface/value_ptr.h"
#include "boost/shared_ptr.hpp"

#include <map>
#include <vector>
#include <cassert>
#include <iosfwd>
#include <set>

namespace edm {

  class RootFile;

  class IndexIntoFile {
    public:
      class IndexIntoFileItr;
      class SortedRunOrLumiItr;
      class IndexRunLumiEventKey;

      typedef long long EntryNumber_t;
      static int const invalidIndex = -1;
      static RunNumber_t const invalidRun = 0U;
      static LuminosityBlockNumber_t const invalidLumi = 0U;
      static EventNumber_t const invalidEvent = 0U;
      static EntryNumber_t const invalidEntry = -1LL;

      enum EntryType {kRun, kLumi, kEvent, kEnd};

      IndexIntoFile();
      ~IndexIntoFile();

      ProcessHistoryID const& processHistoryID(int i) const;
      std::vector<ProcessHistoryID> const& processHistoryIDs() const;

      void addEntry(ProcessHistoryID const& processHistoryID,
                    RunNumber_t run,
                    LuminosityBlockNumber_t lumi,
                    EventNumber_t event,
                    EntryNumber_t entry);

      void fixIndexes(std::vector<ProcessHistoryID>& processHistoryIDs);

      void setNumberOfEvents(EntryNumber_t nevents) const {
        transients_.get().numberOfEvents_ = nevents;
      }

      void sortVector_Run_Or_Lumi_Entries();

      enum SortOrder {numericalOrder, firstAppearanceOrder};

      IndexIntoFileItr begin(SortOrder sortOrder) const;
      IndexIntoFileItr end(SortOrder sortOrder) const;
      bool iterationWillBeInEntryOrder(SortOrder sortOrder) const;

      bool empty() const;

      IndexIntoFileItr
      findPosition(RunNumber_t run, LuminosityBlockNumber_t lumi = 0U, EventNumber_t event = 0U) const;

      IndexIntoFileItr
      findEventPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;

      IndexIntoFileItr
      findLumiPosition(RunNumber_t run, LuminosityBlockNumber_t lumi) const;

      IndexIntoFileItr
      findRunPosition(RunNumber_t run) const;

      bool containsItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;
      bool containsEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;
      bool containsLumi(RunNumber_t run, LuminosityBlockNumber_t lumi) const;
      bool containsRun(RunNumber_t run) const;

      SortedRunOrLumiItr beginRunOrLumi() const;
      SortedRunOrLumiItr endRunOrLumi() const;

      void set_intersection(IndexIntoFile const& indexIntoFile, std::set<IndexRunLumiEventKey>& intersection) const;
      bool containsDuplicateEvents() const;

      void inputFileClosed() const;

      //*****************************************************************************
      //*****************************************************************************

      class RunOrLumiEntry {
      public:

        RunOrLumiEntry();

        RunOrLumiEntry(EntryNumber_t orderPHIDRun,
                       EntryNumber_t orderPHIDRunLumi,
                       EntryNumber_t entry,
                       int processHistoryIDIndex,
                       RunNumber_t run,
                       LuminosityBlockNumber_t lumi,
                       EntryNumber_t beginEvents,
                       EntryNumber_t Event);

        EntryNumber_t orderPHIDRun() const {return orderPHIDRun_;}
        EntryNumber_t orderPHIDRunLumi() const {return orderPHIDRunLumi_;}
        EntryNumber_t entry() const {return entry_;}
        int processHistoryIDIndex() const {return processHistoryIDIndex_;}
        RunNumber_t run() const {return run_;}
        LuminosityBlockNumber_t lumi() const {return lumi_;}
        EntryNumber_t beginEvents() const {return beginEvents_;}
        EntryNumber_t endEvents() const {return endEvents_;}

        bool isRun() const {return lumi() == invalidLumi;}

        void setOrderPHIDRun(EntryNumber_t v) {orderPHIDRun_ = v;}
        void setProcessHistoryIDIndex(int v) {processHistoryIDIndex_ = v;}

        bool operator<(RunOrLumiEntry const& right) const {
          if (orderPHIDRun_ == right.orderPHIDRun()) {
            if (orderPHIDRunLumi_ == right.orderPHIDRunLumi()) {
              return entry_ < right.entry();
            }
            return orderPHIDRunLumi_ < right.orderPHIDRunLumi();
          }
          return orderPHIDRun_ < right.orderPHIDRun();
        }

      private:
        // All Runs, Lumis, and Events associated with the same
        // ProcessHistory and Run in the same input file are processed
        // contiguously.  This parameter establishes the default order
        // of processing of these contiguous subsets of data.
        EntryNumber_t orderPHIDRun_;

        // All Lumis and Events associated with the same
        // ProcessHistory, Run, and Lumi in the same input file are
        // processed contiguously.  This parameter establishes the
        // default order of processing of these contiguous subsets
        // of data which have the same ProcessHistory and Run.
        EntryNumber_t orderPHIDRunLumi_; // -1 if a run

        // TTree entry number of Run or Lumi
        EntryNumber_t entry_;

        int processHistoryIDIndex_;
        RunNumber_t run_;
        LuminosityBlockNumber_t lumi_;  // 0 indicates this is a run entry

        EntryNumber_t beginEvents_;     // -1 if a run or a lumi with no events
        EntryNumber_t endEvents_;       // -1 if a run or a lumi with no events
      };

      //*****************************************************************************
      //*****************************************************************************

      class RunOrLumiIndexes {
      public:
        RunOrLumiIndexes(int processHistoryIDIndex, RunNumber_t run, LuminosityBlockNumber_t lumi, int indexToGetEntry);

        int processHistoryIDIndex() const {return processHistoryIDIndex_;}
      	RunNumber_t run() const {return run_;}
        LuminosityBlockNumber_t lumi() const {return lumi_;}
        int indexToGetEntry() const {return indexToGetEntry_;}
        long long beginEventNumbers() const {return beginEventNumbers_;}
        long long endEventNumbers() const {return endEventNumbers_;}

        bool isRun() const {return lumi() == invalidLumi;}

        void setBeginEventNumbers(long long v) {beginEventNumbers_ = v;}
        void setEndEventNumbers(long long v) {endEventNumbers_ = v;}

        bool operator<(RunOrLumiIndexes const& right) const {
          if (processHistoryIDIndex_ == right.processHistoryIDIndex()) {
            if (run_ == right.run()) {
              return lumi_ < right.lumi();
            }
            return run_ < right.run();
          }
          return processHistoryIDIndex_ < right.processHistoryIDIndex();
        }

      private:

        int processHistoryIDIndex_;
      	RunNumber_t run_;
        LuminosityBlockNumber_t lumi_;    // 0 indicates this is a run entry
        int indexToGetEntry_;

        // The next two data members are indexes into the vectors eventNumbers_ and
        // eventEntries_ (which both have the same number of entries in the same order,
        // the only difference being that one contains only events numbers and is
        // smaller in memory).
        // If there are no events, then the next two are equal (and the value is the
        // index where the first event would have gone if there had been one)
        long long beginEventNumbers_;     // first event this PHID-Run-Lumi (-1 if a run or not set)
        long long endEventNumbers_;       // one past last event this PHID-Run-Lumi (-1 if a run or not set)
      };

      //*****************************************************************************
      //*****************************************************************************

      class EventEntry {
      public:
        EventEntry() : event_(invalidEvent), entry_(invalidEntry) {}
        EventEntry(EventNumber_t event, EntryNumber_t entry) : event_(event), entry_(entry) {}

        EventNumber_t event() const {return event_;}
        EntryNumber_t entry() const {return entry_;}

        bool operator<(EventEntry const& right) const {
          return event() < right.event();
        }
 
        bool operator==(EventEntry const& right) const {
          return event() == right.event();
        }
 
      private:
        EventNumber_t event_;
        EntryNumber_t entry_;
      };


      //*****************************************************************************
      //*****************************************************************************

      class SortedRunOrLumiItr {

      public:
        SortedRunOrLumiItr(IndexIntoFile const* indexIntoFile, unsigned runOrLumi);

	IndexIntoFile const* indexIntoFile() const {return indexIntoFile_;}
        unsigned runOrLumi() const {return runOrLumi_;}

        bool operator==(SortedRunOrLumiItr const& right) const;
        bool operator!=(SortedRunOrLumiItr const& right) const;
        SortedRunOrLumiItr& operator++();

        bool isRun();

        void getRange(long long& beginEventNumbers,
                      long long& endEventNumbers,
                      EntryNumber_t& beginEventEntry,
                      EntryNumber_t& endEventEntry);

	RunOrLumiIndexes const& runOrLumiIndexes() const;

      private:

        IndexIntoFile const* indexIntoFile_;

        // This is an index into runOrLumiIndexes_
        // which gives the current position of the iteration
        unsigned runOrLumi_;
      };


      //*****************************************************************************
      //*****************************************************************************

      class IndexIntoFileItrImpl {

      public:
        IndexIntoFileItrImpl(IndexIntoFile const* indexIntoFile,
                             EntryType entryType,
                             int indexToRun,
                             int indexToLumi,
                             int indexToEventRange,
                             long long indexToEvent,
                             long long nEvents);

        virtual IndexIntoFileItrImpl* clone() const = 0;

	EntryType getEntryType() const {return type_;}

        void next ();

        // Move to whatever is after the current event
        // or next event if there is not a current event,
        // but do not modify the type or run/lumi
        // indexes unless it is necessary because there
	// are no more events in the current run or lumi.
        void skipEventForward(int& phIndexOfSkippedEvent,
                              RunNumber_t& runOfSkippedEvent,
                              LuminosityBlockNumber_t& lumiOfSkippedEvent,
                              EntryNumber_t& skippedEventEntry);

        // Move so that the event immediately preceding the
        // the current position is the next event processed.
        // If the type is kEvent or kLumi, then change the type to kRun
        // if and only if the preceding event is in a different
        // Run. If the type is kEvent, change the type to kLumi if
        // the Lumi is different but the Run is the same.  Otherwise
        // leave the type unchanged.
        void skipEventBackward(int& phIndexOfEvent,
                               RunNumber_t& runOfEvent,
                               LuminosityBlockNumber_t& lumiOfEvent,
                               EntryNumber_t& eventEntry);

        virtual int processHistoryIDIndex() const  = 0;
        virtual RunNumber_t run() const = 0;
        virtual LuminosityBlockNumber_t lumi() const = 0;
        virtual EntryNumber_t entry() const = 0;
        virtual LuminosityBlockNumber_t peekAheadAtLumi() const = 0;
        virtual EntryNumber_t peekAheadAtEventEntry() const = 0;
        virtual bool skipLumiInRun() = 0;

        void advanceToNextRun();
        void advanceToNextLumiOrRun();
        void initializeRun();

        void initializeLumi() {initializeLumi_();}

        bool operator==(IndexIntoFileItrImpl const& right) const;

        IndexIntoFile const* indexIntoFile() const { return indexIntoFile_; }
        int size() const { return size_; }

        EntryType type() const { return type_; }
        int indexToRun() const { return indexToRun_; }

        int indexToLumi() const { return indexToLumi_; }
        int indexToEventRange() const { return indexToEventRange_; }
        long long indexToEvent() const { return indexToEvent_; }
        long long nEvents() const { return nEvents_; }

      protected:

        void setInvalid();

        void setIndexToLumi(int value) { indexToLumi_ = value; }
        void setIndexToEventRange(int value) { indexToEventRange_ = value; }
        void setIndexToEvent(long long value) { indexToEvent_ = value; }
        void setNEvents(long long value) { nEvents_ = value; }

      private:

        virtual void initializeLumi_() = 0;
        virtual bool nextEventRange() = 0;
        virtual bool previousEventRange() = 0;
        bool previousLumiWithEvents();
        virtual bool setToLastEventInRange(int index) = 0;
        virtual EntryType getRunOrLumiEntryType(int index) const = 0;
        virtual bool isSameLumi(int index1, int index2) const = 0;
        virtual bool isSameRun(int index1, int index2) const = 0;

        IndexIntoFile const* indexIntoFile_;
        int size_;

        EntryType type_;
        int indexToRun_;
        int indexToLumi_;
        int indexToEventRange_;
        long long indexToEvent_;
        long long nEvents_;
      };

      //*****************************************************************************
      //*****************************************************************************

      class IndexIntoFileItrNoSort : public IndexIntoFileItrImpl {
      public:
        IndexIntoFileItrNoSort(IndexIntoFile const* indexIntoFile,
                               EntryType entryType,
                               int indexToRun,
                               int indexToLumi,
                               int indexToEventRange,
                               long long indexToEvent,
                               long long nEvents);

        virtual IndexIntoFileItrImpl* clone() const;

        virtual int processHistoryIDIndex() const;
        virtual RunNumber_t run() const;
        virtual LuminosityBlockNumber_t lumi() const;
        virtual EntryNumber_t entry() const;
        virtual LuminosityBlockNumber_t peekAheadAtLumi() const;
        virtual EntryNumber_t peekAheadAtEventEntry() const;
        virtual bool skipLumiInRun();

      private:

        virtual void initializeLumi_();
        virtual bool nextEventRange();
        virtual bool previousEventRange();
        virtual bool setToLastEventInRange(int index);
        virtual EntryType getRunOrLumiEntryType(int index) const;
        virtual bool isSameLumi(int index1, int index2) const;
        virtual bool isSameRun(int index1, int index2) const;
      };

      //*****************************************************************************
      //*****************************************************************************

      class IndexIntoFileItrSorted : public IndexIntoFileItrImpl {
      public:
        IndexIntoFileItrSorted(IndexIntoFile const* indexIntoFile,
                               EntryType entryType,
                               int indexToRun,
                               int indexToLumi,
                               int indexToEventRange,
                               long long indexToEvent,
                               long long nEvents);

        virtual IndexIntoFileItrImpl* clone() const;
        virtual int processHistoryIDIndex() const;
        virtual RunNumber_t run() const;
        virtual LuminosityBlockNumber_t lumi() const;
        virtual EntryNumber_t entry() const;
        virtual LuminosityBlockNumber_t peekAheadAtLumi() const;
        virtual EntryNumber_t peekAheadAtEventEntry() const;
        virtual bool skipLumiInRun();

      private:

        virtual void initializeLumi_();
        virtual bool nextEventRange();
        virtual bool previousEventRange();
        virtual bool setToLastEventInRange(int index);
        virtual EntryType getRunOrLumiEntryType(int index) const;
        virtual bool isSameLumi(int index1, int index2) const;
        virtual bool isSameRun(int index1, int index2) const;
      };

      //*****************************************************************************
      //*****************************************************************************

      class IndexIntoFileItr {
      public:
        IndexIntoFileItr(IndexIntoFile const* indexIntoFile,
                         SortOrder sortOrder,
                         EntryType entryType,
                         int indexToRun,
                         int indexToLumi,
                         int indexToEventRange,
                         long long indexToEvent,
                         long long nEvents);


        EntryType getEntryType() const {return impl_->getEntryType();}
        int processHistoryIDIndex() const {return impl_->processHistoryIDIndex();}
        RunNumber_t run() const {return impl_->run();}
        LuminosityBlockNumber_t lumi() const {return impl_->lumi();}
        EntryNumber_t entry() const {return impl_->entry();}
        LuminosityBlockNumber_t peekAheadAtLumi() const { return impl_->peekAheadAtLumi(); }

        // This is intentionally not implemented.
        // It would be difficult to implement for the no sort mode,
        // either slow or using extra memory.
        // It would be easy to implement for the sorted iteration,
        // but I did not implement it so both cases would offer a
        // consistent interface.
        // It looks like in all cases where this would be needed
        // it would not be difficult to get the event number
        // directly from the event auxiliary.
        // We may need to revisit this decision in the future.
        // EventNumber_t event() const;

        IndexIntoFileItr&  operator++() {
          impl_->next();
          return *this;
        }

        void skipEventForward(int& phIndexOfSkippedEvent,
                              RunNumber_t& runOfSkippedEvent,
                              LuminosityBlockNumber_t& lumiOfSkippedEvent,
                              EntryNumber_t& skippedEventEntry) {
          impl_->skipEventForward(phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, skippedEventEntry);
        }

        void skipEventBackward(int& phIndexOfEvent,
                               RunNumber_t& runOfEvent,
                               LuminosityBlockNumber_t& lumiOfEvent,
                               EntryNumber_t& eventEntry) {
          impl_->skipEventBackward(phIndexOfEvent, runOfEvent, lumiOfEvent, eventEntry);
        }

        bool skipLumiInRun() { return impl_->skipLumiInRun(); }

        void initializeRun() {impl_->initializeRun();}
        void initializeLumi() {impl_->initializeLumi();}
        void advanceToNextRun() {impl_->advanceToNextRun();}
        void advanceToNextLumiOrRun() {impl_->advanceToNextLumiOrRun();}

        void advanceToEvent();
        void advanceToLumi();

        bool operator==(IndexIntoFileItr const& right) const {
          return *impl_ == *right.impl_;
        }

        bool operator!=(IndexIntoFileItr const& right) const {
          return !(*this == right);
        }


      private:

        // The rest of these are intended to be used only by code which tests
        // this class.
        IndexIntoFile const* indexIntoFile() const { return impl_->indexIntoFile(); }
        int size() const { return impl_->size(); }
        EntryType type() const { return impl_->type(); }
        int indexToRun() const { return impl_->indexToRun(); }
        int indexToLumi() const { return impl_->indexToLumi(); }
        int indexToEventRange() const { return impl_->indexToEventRange(); }
        long long indexToEvent() const { return impl_->indexToEvent(); }
        long long nEvents() const { return impl_->nEvents(); }

        value_ptr<IndexIntoFileItrImpl> impl_;
      };

      //*****************************************************************************
      //*****************************************************************************

      class IndexRunKey {
      public:
        IndexRunKey(int index, RunNumber_t run) :
          processHistoryIDIndex_(index),
          run_(run) {
        }

        int processHistoryIDIndex() const {return processHistoryIDIndex_;}
        RunNumber_t run() const {return run_;}

        bool operator<(IndexRunKey const& right) const {
          if (processHistoryIDIndex_ == right.processHistoryIDIndex()) {
            return run_ < right.run();
	  }
          return processHistoryIDIndex_ < right.processHistoryIDIndex();
        }

      private:
        int processHistoryIDIndex_;
        RunNumber_t run_;
      };

      //*****************************************************************************
      //*****************************************************************************

      class IndexRunLumiKey {
      public:
        IndexRunLumiKey(int index, RunNumber_t run, LuminosityBlockNumber_t lumi) :
          processHistoryIDIndex_(index),
          run_(run),
          lumi_(lumi) {
        }

        int processHistoryIDIndex() const {return processHistoryIDIndex_;}
        RunNumber_t run() const {return run_;}
        LuminosityBlockNumber_t lumi() const {return lumi_;}

        bool operator<(IndexRunLumiKey const& right) const {
          if (processHistoryIDIndex_ == right.processHistoryIDIndex()) {
            if (run_ == right.run()) {
              return lumi_ < right.lumi();
            }
            return run_ < right.run();
	  }
          return processHistoryIDIndex_ < right.processHistoryIDIndex();
        }

      private:
        int processHistoryIDIndex_;
        RunNumber_t run_;
        LuminosityBlockNumber_t lumi_;
      };

      //*****************************************************************************
      //*****************************************************************************

      class IndexRunLumiEventKey {
      public:
        IndexRunLumiEventKey(int index, RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) :
          processHistoryIDIndex_(index),
          run_(run),
          lumi_(lumi),
          event_(event) {
        }

        int processHistoryIDIndex() const {return processHistoryIDIndex_;}
        RunNumber_t run() const {return run_;}
        LuminosityBlockNumber_t lumi() const {return lumi_;}
        EventNumber_t event() const {return event_;}

        bool operator<(IndexRunLumiEventKey const& right) const {
          if (processHistoryIDIndex_ == right.processHistoryIDIndex()) {
            if (run_ == right.run()) {
              if (lumi_ == right.lumi()) {
                return event_ < right.event();
              }
              return lumi_ < right.lumi();
            }
            return run_ < right.run();
	  }
          return processHistoryIDIndex_ < right.processHistoryIDIndex();
        }

      private:
        int processHistoryIDIndex_;
        RunNumber_t run_;
        LuminosityBlockNumber_t lumi_;
        EventNumber_t event_;
      };

      //*****************************************************************************
      //*****************************************************************************

      class EventFinder {
      public:
        virtual ~EventFinder() {}
	virtual EventNumber_t getEventNumberOfEntry(EntryNumber_t entry) const = 0;
      };

      //*****************************************************************************
      //*****************************************************************************

      struct Transients {
	Transients();
        int previousAddedIndex_;
        std::map<IndexRunKey, EntryNumber_t> runToFirstEntry_;
        std::map<IndexRunLumiKey, EntryNumber_t> lumiToFirstEntry_;
        EntryNumber_t beginEvents_;
        EntryNumber_t endEvents_;
        int currentIndex_;
        RunNumber_t currentRun_;
        LuminosityBlockNumber_t currentLumi_;
	EntryNumber_t numberOfEvents_;
	boost::shared_ptr<EventFinder> eventFinder_;
        std::vector<RunOrLumiIndexes> runOrLumiIndexes_;
        std::vector<EventNumber_t> eventNumbers_;
        std::vector<EventEntry> eventEntries_;
      };

      //*****************************************************************************
      //*****************************************************************************

      std::vector<RunOrLumiEntry> const& runOrLumiEntries() const {return runOrLumiEntries_;}
      std::vector<RunOrLumiEntry>& setRunOrLumiEntries() {return runOrLumiEntries_;}
      std::vector<ProcessHistoryID>& setProcessHistoryIDs() {return processHistoryIDs_;}
      void setEventFinder(boost::shared_ptr<EventFinder> ptr) const {transients_.get().eventFinder_ = ptr;}
      void fillEventNumbers() const;
      void fillEventEntries() const;

    private:

      void resetEventFinder() const {transients_.get().eventFinder_.reset();}
      std::vector<EventEntry>& eventEntries() const {return transients_.get().eventEntries_;}
      std::vector<EventNumber_t>& eventNumbers() const {return transients_.get().eventNumbers_;}
      void fillRunOrLumiIndexes() const;
      void sortEvents() const;
      void sortEventEntries() const;
      int& previousAddedIndex() const {return transients_.get().previousAddedIndex_;}
      std::map<IndexRunKey, EntryNumber_t>& runToFirstEntry() const {return transients_.get().runToFirstEntry_;}
      std::map<IndexRunLumiKey, EntryNumber_t>& lumiToFirstEntry() const {return transients_.get().lumiToFirstEntry_;}
      EntryNumber_t& beginEvents() const {return transients_.get().beginEvents_;}
      EntryNumber_t& endEvents() const {return transients_.get().endEvents_;}
      int& currentIndex() const {return transients_.get().currentIndex_;}
      RunNumber_t& currentRun() const {return transients_.get().currentRun_;}
      LuminosityBlockNumber_t& currentLumi() const {return transients_.get().currentLumi_;}
      std::vector<RunOrLumiIndexes>& runOrLumiIndexes() const {return transients_.get().runOrLumiIndexes_;}
      size_t numberOfEvents() const {return transients_.get().numberOfEvents_;}
      EventNumber_t getEventNumberOfEntry(EntryNumber_t entry) const {
        return transients_.get().eventFinder_->getEventNumberOfEntry(entry);
      }

      mutable Transient<Transients> transients_;

      std::vector<ProcessHistoryID> processHistoryIDs_;
      std::vector<RunOrLumiEntry> runOrLumiEntries_;
  };

  template <>
  struct value_ptr_traits<IndexIntoFile::IndexIntoFileItrImpl> {
    static IndexIntoFile::IndexIntoFileItrImpl* clone(IndexIntoFile::IndexIntoFileItrImpl const* p) {return p->clone();}
  };


  class Compare_Index_Run {
  public:
    bool operator()(IndexIntoFile::RunOrLumiIndexes const& lh, IndexIntoFile::RunOrLumiIndexes const& rh);
  };

  class Compare_Index {
  public:
    bool operator()(IndexIntoFile::RunOrLumiIndexes const& lh, IndexIntoFile::RunOrLumiIndexes const& rh);
  };
}

#endif
