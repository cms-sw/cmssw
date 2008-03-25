#ifndef FWCore_Framework_NoDelayedReader_h
#define FWCore_Framework_NoDelayedReader_h

/*----------------------------------------------------------------------
$Id: NoDelayedReader.h,v 1.7 2007/05/29 19:28:15 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include "FWCore/Framework/interface/DelayedReader.h"

namespace edm {
  class NoDelayedReader : public DelayedReader {
  public:
    virtual ~NoDelayedReader();
  private:
    virtual std::auto_ptr<EDProduct> getProduct(BranchKey const& k, EDProductGetter const* ep) const;
    virtual std::auto_ptr<BranchEntryDescription> getProvenance(BranchKey const& k) const;
  };
}
#endif
