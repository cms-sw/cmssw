#include "FWCore/FWCoreIntegration/test/OtherThingAlgorithm.h"
#include "FWCore/FWCoreIntegration/test/OtherThing.h"
#include "FWCore/FWCoreIntegration/test/OtherThingCollection.h"
#include "FWCore/FWCoreIntegration/test/ThingCollection.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/Handle.h"

namespace edmreftest {
  void OtherThingAlgorithm::run(edm::Event &e, OtherThingCollection & otherThingCollection) {
    edm::Handle<ThingCollection> things;
    e.getByLabel("Thing", things);
    for (int i = 0; i < 20; ++i) {
      OtherThing tc;
      tc.a = i;
      tc.ref = edm::Ref<ThingCollection>(e.ID(), things->ID(), i);
      tc.refVec.push_back(tc.ref);
      tc.refVec.push_back(edm::Ref<ThingCollection>(e.ID(), things->ID(), 19-i));
      otherThingCollection.push_back(tc);
    }
  }
}
