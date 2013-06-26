#include "FWCore/Integration/test/ThingAlgorithm.h"
#include "DataFormats/TestObjects/interface/Thing.h"

namespace edmtest {
  void ThingAlgorithm::run(ThingCollection & thingCollection) {
    thingCollection.reserve(20);
    for (int i = 0; i < 20; ++i) {
      Thing tc;
      tc.a = i+offset;
      thingCollection.push_back(tc);
    }
    offset += offsetDelta;
  }
}
