#ifndef FWCore_Framework_NoDelayedReader_h
#define FWCore_Framework_NoDelayedReader_h

/*----------------------------------------------------------------------
$Id: NoDelayedReader.h,v 1.6 2007/05/10 12:27:03 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include "FWCore/Framework/interface/DelayedReader.h"

namespace edm {
  struct NoDelayedReader : public DelayedReader {
    virtual ~NoDelayedReader();
    virtual std::auto_ptr<EDProduct> getProduct(BranchKey const& k, EDProductGetter const* ep) const;
    virtual std::auto_ptr<BranchEntryDescription> getProvenance(BranchKey const& k) const;
  };
}
#endif
