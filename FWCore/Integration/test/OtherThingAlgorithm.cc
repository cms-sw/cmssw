#include "FWCore/Integration/test/OtherThingAlgorithm.h"
#include "DataFormats/TestObjects/interface/OtherThing.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

namespace edmtest {
  void OtherThingAlgorithm::run(edm::DataViewImpl const& event, 
				OtherThingCollection& result,
				std::string const& thingLabel, 
				std::string const& instance) {

    const size_t numToMake = 20;
    result.reserve(numToMake);
    edm::Handle<ThingCollection> parent;
    event.getByLabel(thingLabel, instance, parent);
    assert(parent.isValid());

    for (size_t i = 0; i < numToMake; ++i) {
      OtherThing element;
      element.a = i;
      element.refProd = edm::RefProd<ThingCollection>(parent);
      element.ref = edm::Ref<ThingCollection>(element.refProd, i);
      element.ptr = edm::Ptr<Thing>(parent, i);
      element.refVec.push_back(element.ref);
      element.refVec.push_back(element.ref);
      element.refVec.push_back(edm::Ref<ThingCollection>(parent, 19-i));
      edm::RefVector<ThingCollection>::iterator ri = element.refVec.begin();
      element.refVec.erase(ri);
      element.oneNullOneNot.push_back(edm::Ref<ThingCollection>(parent.id()));
      element.oneNullOneNot.push_back(edm::Ref<ThingCollection>(parent, 0));
      element.refToBaseProd = edm::RefToBaseProd<Thing>(element.refProd);
      element.refToBase = edm::RefToBase<Thing>(element.ref);
      assert(element.oneNullOneNot.size() == 2); // we'll check this in our tests
      result.push_back(element);
      //      element.refVec.clear(); // no need to clear; 'element' is created anew on every loop
    }
  }
}
