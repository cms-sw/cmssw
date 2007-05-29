#ifndef DataFormats_Provenance_DelayedReader_h
#define DataFormats_Provenance_DelayedReader_h

/*----------------------------------------------------------------------
  
DelayedReader: The abstract interface through which the EventPrincipal
uses input sources to retrieve per-event provenance from external storage.

$Id: ProvenanceDelayedReader.h,v 1.5 2007/05/10 12:27:03 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

namespace edm {
  class BranchKey;
  class BranchEntryDescription;
  class ProvenanceDelayedReader {
  public:
    virtual ~ProvenanceDelayedReader() {}

    virtual std::auto_ptr<BranchEntryDescription> getProvenance(BranchKey const& k) const = 0;
  };
}

#endif
