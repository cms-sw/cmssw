#ifndef Integration_ThingAlgorithm_h
#define Integration_ThingAlgorithm_h

/** \class ThingAlgorithm
 *
 ************************************************************/
#include "DataFormats/TestObjects/interface/ThingCollectionfwd.h"
#include <atomic>

namespace edmtest {
  class ThingAlgorithm {
  public:
    ThingAlgorithm(long iOffsetDelta = 0, int nThings = 20)
        : offset_(0), offsetDelta_(iOffsetDelta), nThings_(nThings) {}

    /// Runs the algorithm and returns a list of Things
    /// The user declares the vector and calls this method.
    void run(ThingCollection& thingCollection) const;

  private:
    mutable std::atomic<long> offset_;
    const long offsetDelta_;
    const int nThings_;
  };
}

#endif
