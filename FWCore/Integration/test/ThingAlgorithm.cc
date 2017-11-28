#include "FWCore/Integration/test/ThingAlgorithm.h"
#include "DataFormats/TestObjects/interface/Thing.h"

namespace edmtest {
  void ThingAlgorithm::run(ThingCollection& thingCollection) const {
    thingCollection.reserve(nThings_);
    auto offset = offset_.fetch_add(offsetDelta_);
    for (int i = 0; i < nThings_; ++i) {
      Thing tc;
      tc.a = i + offset;
      thingCollection.push_back(tc);
    }
  }
}
