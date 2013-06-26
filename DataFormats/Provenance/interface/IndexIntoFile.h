#ifndef DataFormats_Provenance_IndexIntoFile_h
#define DataFormats_Provenance_IndexIntoFile_h

/** \class edm::IndexIntoFile

Used to quickly find the Events, Lumis, and Runs in a single
ROOT format data file and step through them in the desired
order.

A list of the most important functions that a client would
use directly follows. There are detailed comments below with
the declaration of each function.

The begin and end functions are used to start and stop
an iteration loop. An argument to the iterator constructor
determines the order of iteration.

The functions findPosition, findEventPosition, findRunPosition,
and findLumiPosition are used to navigate directly to specific
runs, lumis, and events.

The functions mentioned above return an object of type
IndexIntoFileItr.  The IndexIntoFileItr class has member
functions which allow one to navigate forward and backward
through the runs, lumis, and events in alternative ways.
See more comments with the declaration of each public member
function in IndexIntoFileItr.

The iterator  will know what the current item is (as one
would expect).  This could be a run, lumi, or event. It
knows more than that though, it knows all three as is
explained below.

In the run state, IndexIntoFileItr knows which lumi will
be processed next after the run and also which event will
be processed after the lumi.  These may not be the first
ones in the run if the skip functions were used.

In the lumi state, the IndexIntoFileItr will always point
at the last associated run and the next event to be processed
after the lumi.  This may not be the first event if the skip
function was used.

In the event state, the IndexIntoFileItr will always point
at the last corresponding run and also the last corresponding
lumi.

There can be multiple run entries in a TTree associated
with the same run number and ProcessHistoryID in a file.
There can also be multiple lumi entries associated with
the same lumi number, run number, and ProcessHistoryID.
Both sorting orders will make these subgroups contiguous,
but beyond that is up to the client (normally PoolSource,
which passes them up to the EventProcessor and EPStates)
to deal with merging the multiple run (or lumi) entries
together.

One final comment with regards to IndexIntoFileItr.  This
is not an STL iterator and it cannot be used with std::
algorithms.  The data structures are complex and designed
to optimize memory usage. It would be difficult or impossible
implement an iterator that is STL compliant.

Here is a summary of the data structures in IndexIntoFile.
The persistent data consists of two vectors.

processHistoryIDs_ is a std::vector<ProcessHistoryID> that
contains the ProcessHistoryIDs with one element in the
vector for each unique ProcessHistoryID. On output they
are ordered as they first written out for each output
file.  On input they are ordered as they are first seen
in each process. Note that each ProcessHistoryID is stored
once in this separate vector. Everywhere else it is needed
it stored as an index into this vector because the
ProcessHistoryID itself is large and it would take more
memory to store them repeatedly in the other vectors.
Note that the ProcessHistoryID's referenced in this
class are always the "reduced" ProcessHistoryID's,
not the ProcessHistoryID of the full ProcessHistory.
You cannot use them to directly access the ProcessHistory
from the ProcessHistoryRegistry.

runOrLumiEntries_ is a std::vector<RunOrLumiEntry>.
This vector holds one element per entry in the run
TTree and one element per entry in the lumi TTree.
When sorted, everything associated with a given run and
ProcessHistoryID will be contiguous in the vector.
These groups of entries will be put in the order they
first appear in the input file. Within each of
these groups the run entries come first in entry order,
followed by the entries associated with the lumis.
The lumis are also contiguous and sorted by first
appearance in the input file. Within a lumi they
are sorted by entry order.

There are a number of transient data members also.
The 3 most important of these are vectors.  To
save memory, these are only filled when needed.

runOrLumiIndexes_ is a std::vector<RunOrLumiIndexes>.
There is a one to one correspondence between the
elements of this vector and the elements of runOrLumiEntries_.
The elements of this vector are sorted in numerical
order using the ProcessHistoryID index, the run number,
and the lumi number. This ordering allows iteration
in numerical order and also fast lookup based on run
number and lumi number. Each element also has indexes
into the eventNumbers_ and eventEntries_ vectors which
hold the information giving the event numbers and
event entry numbers.

eventNumbers_ is a std::vector containing EventNumber_t's.
Each element is a 4 byte int.  eventEntries_ is a
std::vector containing EventEntry's.  Each EventEntry
contains a 4 byte event number and an 8 byte entry number.
If filled, both vectors contain the same number of
entries with identical event numbers sorted in the
same order.  The only difference is that one includes
the entry numbers and thus takes more memory.
Each element of runOrLumiIndexes_ has the indexes necessary
to find the range inside eventNumbers_ or eventEntries_
corresponding to its lumi.  Within that range the elements
are sorted by event number, which is used for the
numerical order iteration and to find an event by the
event number.

The details of the data structure are a little different
when reading files written before release 3_8_0
(backward compatibility, see RootFile::fillIndexIntoFile
for the details).

This data structure is optimized for low memory usage when
there are large numbers of events.  The optimal case would
occur when there was was one run in a file, one luminosity block
in that run and everything had the same ProcessHistoryID.
If duplicate checking were off and the process simply iterated
through the file in the default order, then only the persistent
vectors would be filled.  One vector would contain 2 elements,
one for the run and the other for the lumi. The other
vector would contain one element, the ProcessHistoryID.
Even if there were a billion events, that would be all that
would exist and take up memory.  The data structure is not the
optimal structure for a very sparse skim, but the overheads
should be tolerable given the number of runs and luminosity
blocks that should occur in CMS data.

Normally the only the persistent part of the data structure is
filled in the output module using two functions designed specifically
for that purpose. The functions are addEntry and
sortVector_Run_Or_Lumi_Entries.

There are some complexities associated with filling the data structure,
mostly revolving around optimizations to minimize the per event memory
usage.  The client needs to know which parts of the data structure to
fill. See the functions below named fixIndexes, setNumberOfEvents,
setEventFinder, fillEventNumbers, fillEventEntries, and inputFileClosed.

Note that this class is not intended to be used directly by the average
CMS user.  PoolSource and PoolOutputModule are the main clients.  Other
executables that read ROOT format data files, but do not use PoolSource
may also need to use it directly (FWLite, Fireworks, edmFileUtil ...).
The interface is too complex for general use.

\author W. David Dagenhart, created 19 May, 2010

*/

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/Utilities/interface/value_ptr.h"

#include "boost/shared_ptr.hpp"

#include <cassert>
#include <iosfwd>
#include <map>
#include <set>
#include <vector>

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

      /// This enum is used to specify the order of iteration.
      /// In firstAppearanceOrder there are 3 sort criteria, in order of precedence these are:
      ///
      ///   1. firstAppearance of the ProcessHistoryID and run number in the file
      ///
      ///   2. firstAppearance of the ProcessHistoryID, run number and lumi number in the file
      ///
      ///   3. entry number
      ///
      /// In numerical order the criteria are in order of precedence are:
      ///
      ///   1. processHistoryID index (which are normally in order of appearance in the process)
      ///
      ///   2. run number
      ///
      ///   3. lumi number
      ///
      ///   4. event number
      ///
      ///   5. entry number
      enum SortOrder {numericalOrder, firstAppearanceOrder};

      /// Used to start an iteration over the Runs, Lumis, and Events in a file.
      /// Note the argument specifies the order
      IndexIntoFileItr begin(SortOrder sortOrder) const;

      /// Used to end an iteration over the Runs, Lumis, and Events in a file.
      IndexIntoFileItr end(SortOrder sortOrder) const;

      /// Used to determine whether or not to disable fast cloning.
      bool iterationWillBeInEntryOrder(SortOrder sortOrder) const;

      /// True if no runs, lumis, or events are in the file.
      bool empty() const;

      /// Find a run, lumi, or event.
      /// Returns an iterator pointing at it. The iterator will always
      /// be in numericalOrder mode.
      /// If it is not found the entry type of the iterator will be kEnd.
      /// If it is found the entry type of the iterator will always be
      /// kRun so the next thing to be processed is the run containing
      /// the desired lumi or event or if looking for a run, the run itself.
      /// If the lumi and event arguments are 0 (invalid), then it will
      /// find a run. If only the event argument is 0 (invalid), then
      /// it will find a lumi. If will look for an event if all three
      /// arguments are nonzero or if only the lumi argument is 0 (invalid).
      /// Note that it will find the first match only so if there is more
      /// than one match then the others cannot be found with this method.
      /// The order of the search is by processHistoryID index, then run
      /// number, then lumi number, then event entry.
      /// If searching for a lumi the iterator will advance directly
      /// to the desired lumi after the run even if it is not the
      /// first lumi in the run.  If searching for an event, the
      /// iterator will advance to the lumi containing the run and
      /// then the requested event after run even if there are other
      /// lumis earlier in that run and other events earlier in that lumi.
      IndexIntoFileItr
      findPosition(RunNumber_t run, LuminosityBlockNumber_t lumi = 0U, EventNumber_t event = 0U) const;

      IndexIntoFileItr
      findPosition(SortOrder sortOrder, RunNumber_t run, LuminosityBlockNumber_t lumi = 0U, EventNumber_t event = 0U) const;

      /// Same as findPosition,except the entry type of the returned iterator will be kEvent or kEnd and the event argument must be nonzero.
      /// This means the next thing to be processed will be the event if it is found.
      IndexIntoFileItr
      findEventPosition(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;

      /// Same as findPosition,except the entry type of the returned iterator will be kLumi or kEnd and the lumi argument must be nonzero.
      /// This means the next thing to be processed will be the lumi if it is found.
      IndexIntoFileItr
      findLumiPosition(RunNumber_t run, LuminosityBlockNumber_t lumi) const;

      /// Same as findPosition.
      IndexIntoFileItr
      findRunPosition(RunNumber_t run) const;

      bool containsItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;
      bool containsEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;
      bool containsLumi(RunNumber_t run, LuminosityBlockNumber_t lumi) const;
      bool containsRun(RunNumber_t run) const;

      SortedRunOrLumiItr beginRunOrLumi() const;
      SortedRunOrLumiItr endRunOrLumi() const;

      /// The intersection argument will be filled with an entry for each event in both IndexIntoFile objects.
      /// To be added the event must have the same ProcessHistoryID index, run number, lumi number and event number.
      void set_intersection(IndexIntoFile const& indexIntoFile, std::set<IndexRunLumiEventKey>& intersection) const;

      /// Returns true if the IndexIntoFile contains 2 events with the same ProcessHistoryID index, run number, lumi number and event number.
      bool containsDuplicateEvents() const;

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
        void setOrderPHIDRunLumi(EntryNumber_t v) {orderPHIDRunLumi_ = v;}
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

        // These are entry numbers in the Events TTree
        // Each RunOrLumiEntry is associated with one contiguous range of events.
        // This is disjoint from the ranges associated with all other RunOrLumiEntry's
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

        // Note that there can be many RunOrLumiIndexes objects where these two values are
        // the same if there are many noncontiguous ranges of events associated with the same
        // PHID-Run-Lumi (this one range in eventNumbers_ corresponds to the union of
        // all the noncontiguous ranges in the Events TTree).
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
         virtual ~IndexIntoFileItrImpl();

        virtual IndexIntoFileItrImpl* clone() const = 0;

        EntryType getEntryType() const {return type_;}

        void next ();

        void skipEventForward(int& phIndexOfSkippedEvent,
                              RunNumber_t& runOfSkippedEvent,
                              LuminosityBlockNumber_t& lumiOfSkippedEvent,
                              EntryNumber_t& skippedEventEntry);

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
        EntryNumber_t firstEventEntryThisRun();
        EntryNumber_t firstEventEntryThisLumi();
        virtual bool skipLumiInRun() = 0;

        void advanceToNextRun();
        void advanceToNextLumiOrRun();
        bool skipToNextEventInLumi();
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

        void copyPosition(IndexIntoFileItrImpl const& position);

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
        /// This itended to be used only internally and by IndexIntoFile.
        /// One thing that is needed for the future, is to add some checks
        /// to make sure the iterator is in a valid state inside this constructor.
        /// It is currently possible to create an iterator with this constructor
        /// in an invalid state and the behavior would then be undefined. In the
        /// existing internal usages the iterator will always be valid.  (for
        /// example IndexIntoFile::begin and IndexIntoFile::findPosition will
        /// always return a valid iterator).
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

        /// Same as lumi() except when the the current type is kRun.
        /// In that case instead of always returning 0 (invalid), it will return the lumi that will be processed next
        LuminosityBlockNumber_t peekAheadAtLumi() const { return impl_->peekAheadAtLumi(); }

        /// Same as entry() except when the the current type is kRun or kLumi.
        /// In that case instead of always returning -1 (invalid), it will return
        /// the event entry that will be processed next and which is in the current
        /// run and lumi. If there is none it still returns -1 (invalid).
        EntryNumber_t peekAheadAtEventEntry() const { return impl_->peekAheadAtEventEntry(); }

        /// Returns the TTree entry of the first event which would be processed in the
        /// current run/lumi if all the events in the run/lumi were processed in the
        /// current processing order. If there are none it returns -1 (invalid).
        EntryNumber_t firstEventEntryThisRun() const { return impl_->firstEventEntryThisRun(); }
        EntryNumber_t firstEventEntryThisLumi() const { return impl_->firstEventEntryThisLumi(); }

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


        /// Move to next event to be processed
        IndexIntoFileItr&  operator++() {
          impl_->next();
          return *this;
        }

        /// Move to whatever is immediately after the current event
        /// or after the next event if there is not a current event,
        /// but do not modify the type or run/lumi
        /// indexes unless it is necessary because there
        /// are no more events in the current run or lumi.
        void skipEventForward(int& phIndexOfSkippedEvent,
                              RunNumber_t& runOfSkippedEvent,
                              LuminosityBlockNumber_t& lumiOfSkippedEvent,
                              EntryNumber_t& skippedEventEntry) {
          impl_->skipEventForward(phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, skippedEventEntry);
        }

        /// Move so that the event immediately preceding the
        /// the current position is the next event processed.
        /// If the type is kEvent or kLumi, then change the type to kRun
        /// if and only if the preceding event is in a different
        /// run. If the type is kEvent, change the type to kLumi if
        /// the lumi is different but the run is the same.  Otherwise
        /// leave the type unchanged.
        void skipEventBackward(int& phIndexOfEvent,
                               RunNumber_t& runOfEvent,
                               LuminosityBlockNumber_t& lumiOfEvent,
                               EntryNumber_t& eventEntry) {
          impl_->skipEventBackward(phIndexOfEvent, runOfEvent, lumiOfEvent, eventEntry);
        }

        /// Move to the next lumi in the current run.
        /// Returns false if there is not one.
        bool skipLumiInRun() { return impl_->skipLumiInRun(); }

        /// Move to the next event in the current lumi.
        /// Returns false if there is not one.
        bool skipToNextEventInLumi() { return impl_->skipToNextEventInLumi(); }

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

        /// Should only be used internally and for tests
        void initializeRun() {impl_->initializeRun();}

        /// Should only be used internally and for tests
        void initializeLumi() {impl_->initializeLumi();}

        /// Copy the position without modifying the pointer to the IndexIntoFile or size
        void copyPosition(IndexIntoFileItr const& position);

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

      // The next two functions are used by the output module to fill the
      // persistent data members.

      /// Used by RootOutputModule to fill the persistent data.
      /// This will not work properly if entries are not added in the same order as in RootOutputModule
      void addEntry(ProcessHistoryID const& processHistoryID,
                    RunNumber_t run,
                    LuminosityBlockNumber_t lumi,
                    EventNumber_t event,
                    EntryNumber_t entry);

      /// Used by RootOutputModule after all entries have been added.
      /// This only works after the correct sequence of addEntry calls,
      /// because it makes some corrections before sorting.  A std::stable_sort
      /// works in cases where those corrections are not needed.
      void sortVector_Run_Or_Lumi_Entries();

      //*****************************************************************************
      //*****************************************************************************

      // The next group of functions is used by the PoolSource (or other
      // input related code) to fill the IndexIntoFile.

      /// Used by PoolSource to force the ProcessHistoryID indexes to be consistent across all input files.
      /// Currently this consistency is important when duplicate checking across all input files.
      /// It may be important for other reasons in the future.
      /// It is important this be called immediately after reading in the object from the input file,
      /// before filling the transient data members or using the indexes in any way.
      void fixIndexes(std::vector<ProcessHistoryID>& processHistoryIDs);

      /// The number of events needs to be set before filling the transient event vectors.
      /// It is used to resize them.
      void setNumberOfEvents(EntryNumber_t nevents) const {
        transient_.numberOfEvents_ = nevents;
      }

      /// Calling this enables the functions that fill the event vectors to get the event numbers.
      /// It needs to be called before filling the events vectors
      /// This implies the client needs to define a class that inherits from
      /// EventFinder and then create one.  This function is used to pass in a
      /// pointer to its base class.
      void setEventFinder(boost::shared_ptr<EventFinder> ptr) const {transient_.eventFinder_ = ptr;}

      /// Fills a vector of 4 byte event numbers.
      /// Not filling it reduces the memory used by IndexIntoFile.
      /// As long as the event finder is still pointing at an open file
      /// this will automatically be called on demand (when the event
      /// numbers are are needed). In cases, where the input file may be
      /// closed when the need arises, the client code must call this
      /// explicitly and fill the vector before the file is closed.
      /// In PoolSource, this is necessary when duplicate checking across
      /// all files and when doing lookups to see if an event is in a
      /// previously opened file.  Either this vector or the one that
      /// also contains event entry numbers can be used when looking for
      /// duplicate events within the same file or looking up events in
      /// in the current file without reading them.
      void fillEventNumbers() const;

      /// Fills a vector of objects that contain a 4 byte event number and
      /// the corresponding TTree entry number (8 bytes) for the event.
      /// Not filling it reduces the memory used by IndexIntoFile.
      /// As long as the event finder is still pointing at an open file
      /// this will automatically be called on demand (when the event
      /// numbers and entries are are needed).  It makes sense for the
      /// client to fill this explicitly in advance if it is known that
      /// it will be needed, because in some cases this will prevent the
      /// event numbers vector from being unnecessarily filled (wasting
      /// memory).  This vector will be needed when iterating over events
      /// in numerical order or looking up specific events. The entry
      /// numbers are needed if the events are actually read from the
      /// input file.
      void fillEventEntries() const;

      /// If needEventNumbers is true then this function does the same thing
      /// as fillEventNumbers.  If NeedEventEntries is true, then this function
      /// does the same thing as fillEventEntries.  If both are true, it fills
      /// both within the same loop and it uses less CPU than calling those
      /// two functions separately.
      void fillEventNumbersOrEntries(bool needEventNumbers, bool needEventEntries) const;

      /// If something external to IndexIntoFile is reading through the EventAuxiliary
      /// then it could use this to fill in the event numbers so that IndexIntoFile
      /// will not read through it again.
      std::vector<EventNumber_t>& unsortedEventNumbers() const {return transient_.unsortedEventNumbers_;}

      /// Clear some vectors and eventFinder when an input file is closed.
      /// This reduces the memory used by IndexIntoFile
      void inputFileClosed() const;

      /// Clears the temporary vector of event numbers to reduce memory usage
      void doneFileInitialization() const;

      /// Used for backward compatibility and tests.
      /// RootFile::fillIndexIntoFile uses this to deal with input files created
      /// with releases before 3_8_0 which do not contain an IndexIntoFile.
      std::vector<RunOrLumiEntry>& setRunOrLumiEntries() {return runOrLumiEntries_;}

      /// Used for backward compatibility and tests.
      /// RootFile::fillIndexIntoFile uses this to deal with input files created
      /// with releases before 3_8_0 which do not contain an IndexIntoFile.
      std::vector<ProcessHistoryID>& setProcessHistoryIDs() {return processHistoryIDs_;}

      /// Used for backward compatibility to convert objects created with releases
      /// that used the full ProcessHistoryID in IndexIntoFile to use the reduced
      /// ProcessHistoryID.
      void reduceProcessHistoryIDs();

      //*****************************************************************************
      //*****************************************************************************

      /// Used internally and for test purposes.
      std::vector<RunOrLumiEntry> const& runOrLumiEntries() const {return runOrLumiEntries_;}

      //*****************************************************************************
      //*****************************************************************************

      void initializeTransients() const {transient_.reset();}

      struct Transients {
        Transients();
        void reset();
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
        std::vector<EventNumber_t> unsortedEventNumbers_;
      };

    private:

      /// This function will automatically get called when needed.
      /// It depends only on the fact that the persistent data has been filled already.
      void fillRunOrLumiIndexes() const;

      void fillUnsortedEventNumbers() const;
      void resetEventFinder() const {transient_.eventFinder_.reset();}
      std::vector<EventEntry>& eventEntries() const {return transient_.eventEntries_;}
      std::vector<EventNumber_t>& eventNumbers() const {return transient_.eventNumbers_;}
      void sortEvents() const;
      void sortEventEntries() const;
      int& previousAddedIndex() const {return transient_.previousAddedIndex_;}
      std::map<IndexRunKey, EntryNumber_t>& runToFirstEntry() const {return transient_.runToFirstEntry_;}
      std::map<IndexRunLumiKey, EntryNumber_t>& lumiToFirstEntry() const {return transient_.lumiToFirstEntry_;}
      EntryNumber_t& beginEvents() const {return transient_.beginEvents_;}
      EntryNumber_t& endEvents() const {return transient_.endEvents_;}
      int& currentIndex() const {return transient_.currentIndex_;}
      RunNumber_t& currentRun() const {return transient_.currentRun_;}
      LuminosityBlockNumber_t& currentLumi() const {return transient_.currentLumi_;}
      std::vector<RunOrLumiIndexes>& runOrLumiIndexes() const {return transient_.runOrLumiIndexes_;}
      size_t numberOfEvents() const {return transient_.numberOfEvents_;}
      EventNumber_t getEventNumberOfEntry(EntryNumber_t entry) const {
        return transient_.eventFinder_->getEventNumberOfEntry(entry);
      }

      mutable Transients transient_;

      std::vector<ProcessHistoryID> processHistoryIDs_; // of reduced process histories
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
