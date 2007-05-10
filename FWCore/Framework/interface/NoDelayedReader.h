#ifndef Framework_NoDelayedReader_h
#define Framework_NoDelayedReader_h

/*----------------------------------------------------------------------
$Id: NoDelayedReader.h,v 1.5 2007/01/23 00:31:05 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include "FWCore/Framework/interface/DelayedReader.h"

namespace edm {
  struct NoDelayedReader : public DelayedReader {
    virtual ~NoDelayedReader();
    virtual std::auto_ptr<EDProduct> getProduct(BranchKey const& k, EDProductGetter const* ep) const;
    virtual std::auto_ptr<BranchEntryDescription> getProvenance(BranchKey const& k, EDProductGetter const* ep) const;
  };
}
#endif
