#include "FWCore/Integration/test/ThingAlgorithm.h"
#include "DataFormats/TestObjects/interface/Thing.h"

namespace edmtest {
  void ThingAlgorithm::run(ThingCollection& thingCollection) const {
    thingCollection.reserve(nThings_);
    auto offset = offset_.fetch_add(offsetDelta_);
    int nItems = nThings_;
    if (grow_) {
      nItems *= offset;
    }
    for (int i = 0; i < nItems; ++i) {
      Thing tc;
      tc.a = i + offset;
      thingCollection.push_back(tc);
    }
  }
}  // namespace edmtest
