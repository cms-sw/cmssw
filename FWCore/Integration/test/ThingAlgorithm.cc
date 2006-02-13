#include "FWCore/Integration/test/ThingAlgorithm.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"

namespace edmtest {
  void ThingAlgorithm::run(ThingCollection & thingCollection) {
    for (int i = 0; i < 20; ++i) {
      Thing tc;
      tc.a = i;
      thingCollection.push_back(tc);
    }
  }
}
