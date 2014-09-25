#ifndef Integration_ThingAlgorithm_h
#define Integration_ThingAlgorithm_h

/** \class ThingAlgorithm
 *
 ************************************************************/
#include "DataFormats/TestObjects/interface/ThingCollectionfwd.h"

namespace edmtest {
  class ThingAlgorithm {
  public:
  ThingAlgorithm(long iOffsetDelta = 0, int nThings = 20) : theDebugLevel(0), offset(0), offsetDelta(iOffsetDelta),
      nThings_(nThings) {}

    /// Runs the algorithm and returns a list of Things
    /// The user declares the vector and calls this method.
    void run(ThingCollection& thingCollection);
  private:
    int    theDebugLevel;
    long   offset;
    long   offsetDelta;
    int    nThings_;
  };

}

#endif
