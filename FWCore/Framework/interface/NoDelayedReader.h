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
    virtual WrapperHolder getProduct_(BranchKey const& k, WrapperInterfaceBase const* interface, EDProductGetter const* ep) const;
    virtual void reset_() {}
  };
}
#endif
