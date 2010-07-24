#ifndef FWCore_Framework_NoDelayedReader_h
#define FWCore_Framework_NoDelayedReader_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <memory>
#include "FWCore/Framework/interface/DelayedReader.h"

namespace edm {
  class NoDelayedReader : public DelayedReader {
  public:
    virtual ~NoDelayedReader();
  private:
    virtual std::auto_ptr<EDProduct> getProduct_(BranchKey const& k, EDProductGetter const* ep) const;
  };
}
#endif
