#include "FWCore/Integration/test/ThingAlgorithm.h"
#include "FWCore/Integration/test/Thing.h"
#include "FWCore/Integration/test/ThingCollection.h"

namespace edmreftest {
  void ThingAlgorithm::run(ThingCollection & thingCollection) {
    for (int i = 0; i < 20; ++i) {
      Thing tc;
      tc.a = i;
      thingCollection.push_back(tc);
    }
  }
}
