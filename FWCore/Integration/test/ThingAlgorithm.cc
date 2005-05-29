#include "FWCore/FWCoreIntegration/test/ThingAlgorithm.h"
#include "FWCore/FWCoreIntegration/test/Thing.h"
#include "FWCore/FWCoreIntegration/test/ThingCollection.h"

namespace edmreftest {
  void ThingAlgorithm::run(ThingCollection & thingCollection) {
    for (int i = 0; i < 20; ++i) {
      Thing tc;
      tc.a = i;
      thingCollection.push_back(tc);
    }
  }
}
