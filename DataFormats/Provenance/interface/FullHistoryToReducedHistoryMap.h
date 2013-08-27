#ifndef DataFormats_Provenance_FullHistoryToReducedHistoryMap_h
#define DataFormats_Provenance_FullHistoryToReducedHistoryMap_h

/** \class edm::IndexIntoFile

Used to convert the ProcessHistoryID of a full ProcessHistory
to the ProcessHistoryID of the corresponding reduced ProcessHistory.

The ProcessHistoryRegistry includes an instance of this class
as its "extra" data member.
An entry will be added when an entry is added to the registry.

\author W. David Dagenhart, created 2 August, 2011
\author Bill Tanenbaum, modified 22 August, 2013 
*/

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include <map>

namespace edm {

  class ProcessHistory;
  class FullHistoryToReducedHistoryMap {
  public:
    FullHistoryToReducedHistoryMap();
#ifndef __GCCXML__
    FullHistoryToReducedHistoryMap(FullHistoryToReducedHistoryMap const&) = delete; // Disallow copying and moving
    FullHistoryToReducedHistoryMap& operator=(FullHistoryToReducedHistoryMap const&) = delete; // Disallow copying and moving
#endif
    /// Use to obtain reduced ProcessHistoryID's from full ProcessHistoryID's
    ProcessHistoryID const& reducedProcessHistoryID(ProcessHistoryID const& fullID) const;
    void insertMapping(ProcessHistory const& processHistory); 

  private:
    typedef std::map<ProcessHistoryID, ProcessHistoryID> Map;
    Map fullHistoryToReducedHistoryMap_;
  };
}
#endif
