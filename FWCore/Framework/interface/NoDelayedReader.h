#ifndef Framework_NoDelayedReader_h
#define Framework_NoDelayedReader_h

/*----------------------------------------------------------------------
$Id: NoDelayedReader.h,v 1.4 2006/01/06 00:29:32 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include "FWCore/Framework/interface/DelayedReader.h"

namespace edm {
  struct NoDelayedReader : public DelayedReader {
    virtual ~NoDelayedReader();
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k, EDProductGetter const* ep) const;
  };
}
#endif
