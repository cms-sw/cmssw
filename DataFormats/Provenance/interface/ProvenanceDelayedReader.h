#ifndef DataFormats_Provenance_DelayedReader_h
#define DataFormats_Provenance_DelayedReader_h

/*----------------------------------------------------------------------
  
DelayedReader: The abstract interface through which the EventPrincipal
uses input sources to retrieve per-event provenance from external storage.

$Id: ProvenanceDelayedReader.h,v 1.3 2008/01/30 00:17:51 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include "DataFormats/Provenance/interface/EntryDescription.h"

namespace edm {
  class BranchKey;
  class EntryDescription;
  class ProvenanceDelayedReader {
  public:
    virtual ~ProvenanceDelayedReader();
    std::auto_ptr<EntryDescription> getProvenance(BranchKey const& k) const {
      return getProvenance_(k);
    }
 private: 
    virtual std::auto_ptr<EntryDescription> getProvenance_(BranchKey const& k) const = 0;
  };
}

#endif
