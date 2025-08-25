#ifndef DataFormats_Provenance_processingOrderMerge_h
#define DataFormats_Provenance_processingOrderMerge_h

#include <string>
#include <vector>

namespace edm {
  class ProcessHistory;
  /**
   * Merge the processing order from the given ProcessHistory into the
   * given vector of process names. Will throw if ordering in processNames is not compatible.
   * The vector will be filled with process names in reverse time order (most recent to oldest).
   */
  void processingOrderMerge(ProcessHistory const& iHistory, std::vector<std::string>& processNames);
  /**
     * Merge the processing order from the given vector of process names into the
     * given vector of process names. Will throw if ordering in processNames is not compatible.
     * The vectors must both be in reverse time order (most recent to oldest).
     */
  void processingOrderMerge(std::vector<std::string> const& iFrom, std::vector<std::string>& iTo);
}  // namespace edm
#endif
