#ifndef DataFormats_Provenance_FullHistoryToReducedHistoryMap_h
#define DataFormats_Provenance_FullHistoryToReducedHistoryMap_h

/** \class edm::IndexIntoFile

Used to convert the ProcessHistoryID of a full ProcessHistory
to the ProcessHistoryID of the corresponding reduced ProcessHistory.

Includes optimizations to cache the result so the same conversion
need not be repeated many times.

The ProcessHistoryRegistry includes an instance of this class
as its "extra" data member.  That instance should be used.
It would be a waste of memory and cpu to instantiate other
instances of this class. The syntax should look something
like the following:

  edm::ProcessHistoryID reducedPHID = edm::ProcessHistoryRegistry::instance()->extra().reduceProcessHistoryID(fullID);

Note that the above function will throw an exception if the
full ProcessHistory is not already in the ProcessHistoryRegistry.
We expect that the reduced ProcessHistory will not ever be
in the registry (although there is nothing prevents that).

\author W. David Dagenhart, created 2 August, 2011

*/

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include <map>

namespace edm {

  class FullHistoryToReducedHistoryMap {
  public:

    FullHistoryToReducedHistoryMap();

    /// Use to obtain reduced ProcessHistoryID's from full ProcessHistoryID's
    ProcessHistoryID const& reduceProcessHistoryID(ProcessHistoryID const& fullID);

  private:
    FullHistoryToReducedHistoryMap(FullHistoryToReducedHistoryMap const&);
    FullHistoryToReducedHistoryMap& operator=(FullHistoryToReducedHistoryMap const&);

    typedef std::map<ProcessHistoryID, ProcessHistoryID> Map;
    Map cache_;
    Map::const_iterator previous_;
  };
}
#endif
