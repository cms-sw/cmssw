#include "FWCore/Integration/test/OtherThingAlgorithm.h"
#include "DataFormats/TestObjects/interface/OtherThing.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

namespace edmtest {
  void OtherThingAlgorithm::run(const edm::Event &e, OtherThingCollection & otherThingCollection) {
    edm::Handle<ThingCollection> things;
    e.getByLabel("Thing", things);
    for (int i = 0; i < 20; ++i) {
      OtherThing tc;
      tc.a = i;
      tc.refProd = edm::RefProd<ThingCollection>(things);
      // tc.ref = edm::Ref<ThingCollection>(things, i);
      tc.ref = edm::Ref<ThingCollection>(tc.refProd, i);
      tc.refVec.push_back(tc.ref);
      tc.refVec.push_back(tc.ref);
      tc.refVec.push_back(edm::Ref<ThingCollection>(things, 19-i));
      edm::RefVector<ThingCollection>::iterator ri = tc.refVec.begin();
      tc.refVec.erase(ri);
      otherThingCollection.push_back(tc);
      tc.refVec.clear();
    }
  }
}
