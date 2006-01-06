#ifndef Framework_NoDelayedReader_h
#define Framework_NoDelayedReader_h

/*----------------------------------------------------------------------
$Id: NoDelayedReader.h,v 1.3 2005/12/28 00:14:52 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/DelayedReader.h"

namespace edm {
  struct NoDelayedReader : public DelayedReader {
    virtual ~NoDelayedReader();
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k, EDProductGetter const* ep) const;
  };
}
#endif
