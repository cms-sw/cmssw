#ifndef FWCore_Framework_NoDelayedReader_h
#define FWCore_Framework_NoDelayedReader_h

/*----------------------------------------------------------------------
$Id: NoDelayedReader.h,v 1.9 2008/01/30 00:32:01 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include "FWCore/Framework/interface/DelayedReader.h"

namespace edm {
  class NoDelayedReader : public DelayedReader {
  public:
    virtual ~NoDelayedReader();
  private:
    virtual std::auto_ptr<EDProduct> getProduct_(BranchKey const& k, EDProductGetter const* ep) const;
    virtual std::auto_ptr<EntryDescription> getProvenance_(BranchKey const& k) const;
  };
}
#endif
