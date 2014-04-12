#ifndef DataFormats_Provenance_ProcessHistoryRegistry_h
#define DataFormats_Provenance_ProcessHistoryRegistry_h

/** \class edm::ProcessHistoryRegistry
\author Bill Tanenbaum, modified 23 August, 2013 
*/

#include <map>
#include <vector>

#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

namespace edm {
  typedef std::map<ProcessHistoryID, ProcessHistory> ProcessHistoryMap;
  typedef std::vector<ProcessHistory> ProcessHistoryVector;

  class ProcessHistoryRegistry {
  public:
    typedef ProcessHistory value_type;
    typedef ProcessHistoryMap collection_type;
    typedef ProcessHistoryVector vector_type;

    ProcessHistoryRegistry();
#ifndef __GCCXML__
    ProcessHistoryRegistry(ProcessHistoryRegistry const&) = delete; // Disallow copying and moving
    ProcessHistoryRegistry& operator=(ProcessHistoryRegistry const&) = delete; // Disallow copying and moving
#endif
    bool registerProcessHistory(ProcessHistory const& processHistory);
    bool getMapped(ProcessHistoryID const& key, ProcessHistory& value) const;
    ProcessHistory const* getMapped(ProcessHistoryID const& key) const;
    ProcessHistoryID const& reducedProcessHistoryID(ProcessHistoryID const& fullID) const;
    ProcessHistoryMap::const_iterator begin() const {
      return data_.begin();
    }
    ProcessHistoryMap::const_iterator end() const {
      return data_.end();
    }
  private:
    ProcessHistoryMap data_;
    std::map<ProcessHistoryID, ProcessHistoryID> extra_;
  };
}
#endif
