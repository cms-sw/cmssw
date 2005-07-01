#ifndef EDMREFTEST_THINGALGORITHM_H
#define EDMREFTEST_THINGALGORITHM_H

/** \class ThingAlgorithm
 *
 ************************************************************/


#include <vector>

namespace edmreftest {
  class Thing;
  typedef std::vector<Thing> ThingCollection;

  class ThingAlgorithm {
  public:
    ThingAlgorithm() : theDebugLevel(0) {}

    /// Runs the algorithm and returns a list of Things
    /// The user declares the vector and calls this method.
    void run(ThingCollection& thingCollection);
  private:
    int    theDebugLevel;
  };

}

#endif
