#ifndef Framework_NoDelayedReader_h
#define Framework_NoDelayedReader_h

/*----------------------------------------------------------------------
$Id: NoDelayedReader.h,v 1.2 2005/12/01 22:14:54 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/DelayedReader.h"

namespace edm {
  struct NoDelayedReader : public DelayedReader {
    virtual ~NoDelayedReader();
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k, EventPrincipal const* ep) const;
  };
}
#endif
