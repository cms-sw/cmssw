#ifndef FWCore_Framework_HistoryAppender_h
#define FWCore_Framework_HistoryAppender_h

#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include <memory>

namespace edm {

  class ProcessConfiguration;
  class ProcessHistoryRegistry;

  class HistoryAppender {
  public:

    HistoryAppender();

    // Used to append the current process to the process history
    // when necessary. Optimized to cache the results so it
    // does not need to repeat the same calculations many times.
    std::shared_ptr<ProcessHistory const>
    appendToProcessHistory(ProcessHistoryID const& inputPHID,
                           ProcessHistory const* inputProcessHistory,
                           ProcessConfiguration const& pc);

  private:
    HistoryAppender(HistoryAppender const&) = delete;
    HistoryAppender& operator=(HistoryAppender const&) = delete;

    // Throws if the new process name is already in the process
    // process history
    void checkProcessHistory(ProcessHistory const& ph,
                             ProcessConfiguration const& pc) const;

    ProcessHistoryID m_cachedInputPHID;
    std::shared_ptr<ProcessHistory const> m_cachedHistory;
  };
}
#endif
