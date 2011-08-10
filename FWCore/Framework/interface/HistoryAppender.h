#ifndef FWCore_Framework_HistoryAppender_h
#define FWCore_Framework_HistoryAppender_h

#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include <map>

namespace edm {

  class ProcessConfiguration;

  class CachedHistory {
  public:

    CachedHistory(ProcessHistory const* inputProcessHistory,
                  ProcessHistory const* processHistory,
                  ProcessHistoryID const& processHistoryID) :
      inputProcessHistory_(inputProcessHistory),
      processHistory_(processHistory),
      processHistoryID_(processHistoryID) {
    }

    ProcessHistory const* inputProcessHistory() const { return inputProcessHistory_; }
    ProcessHistory const* processHistory() const { return processHistory_; }
    ProcessHistoryID const& processHistoryID () const { return processHistoryID_; }

  private:

    // This class does not own the memory
    ProcessHistory const* inputProcessHistory_;
    ProcessHistory const* processHistory_;

    ProcessHistoryID processHistoryID_;
  };

  class HistoryAppender {
  public:

    HistoryAppender();

    // Used to append the current process to the process history
    // when necessary. Optimized to cache the results so it
    // does not need to repeat the same calculations many times.
    CachedHistory const&
    appendToProcessHistory(ProcessHistoryID const& inputPHID,
                           ProcessConfiguration const& pc);

  private:
    HistoryAppender(HistoryAppender const&);
    HistoryAppender& operator=(HistoryAppender const&);

    // Throws if the new process name is already in the process
    // process history
    void checkProcessHistory(ProcessHistory const& ph,
                             ProcessConfiguration const& pc) const;

    // The map is intended to have the key be the ProcessHistoryID
    // read from the input file in one of the Auxiliary objects.
    // The CachedHistory has the ProcessHistoryID after adding
    // the current process and the two pointers to the corresponding
    // ProcessHistory objects in the registry, except if the history
    // is empty then the pointer is to the data member of this class
    // because the empty one is never in the registry.
    typedef std::map<ProcessHistoryID, CachedHistory> HistoryMap;
    HistoryMap historyMap_;

    // We cache iterator to the previous element for
    // performance. We expect the IDs to repeat many times
    // and this avoids the lookup in that case.
    HistoryMap::const_iterator previous_;

    ProcessHistory emptyHistory_;
  };
}
#endif
