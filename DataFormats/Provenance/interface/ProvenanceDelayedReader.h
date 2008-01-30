#ifndef DataFormats_Provenance_DelayedReader_h
#define DataFormats_Provenance_DelayedReader_h

/*----------------------------------------------------------------------
  
DelayedReader: The abstract interface through which the EventPrincipal
uses input sources to retrieve per-event provenance from external storage.

$Id: ProvenanceDelayedReader.h,v 1.2 2007/08/15 22:41:05 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

namespace edm {
  class BranchKey;
  class EntryDescription;
  class ProvenanceDelayedReader {
  public:
    virtual ~ProvenanceDelayedReader();
    virtual std::auto_ptr<EntryDescription> getProvenance(BranchKey const& k) const = 0;
  };
}

#endif
