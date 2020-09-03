#include "FWCore/Integration/test/OtherThingAlgorithm.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TestObjects/interface/OtherThing.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToPtr.h"

namespace edmtest {
  void OtherThingAlgorithm::run(edm::Handle<ThingCollection> const& parentHandle,
                                OtherThingCollection& result,
                                bool useRefs,
                                bool refsAreTransient) const {
    const size_t numToMake = 20;
    result.reserve(numToMake);
    ThingCollection const* null = 0;
    ThingCollection const* parent = nullptr;
    if (useRefs) {
      parent = parentHandle.product();
    }
    for (size_t i = 0; i < numToMake; ++i) {
      OtherThing element;
      if (!useRefs) {
        element.a = i;
      } else if (refsAreTransient) {
        element.a = i;
        element.refProd = edm::RefProd<ThingCollection>(parent);
        element.ref = edm::Ref<ThingCollection>(element.refProd, i);
        element.refVec.push_back(element.ref);
        element.refVec.push_back(element.ref);
        element.refVec.push_back(edm::Ref<ThingCollection>(parent, 19 - i));
        edm::RefVector<ThingCollection>::iterator ri = element.refVec.begin();
        element.refVec.erase(ri);
        element.oneNullOneNot.push_back(edm::Ref<ThingCollection>(null, 0));
        element.oneNullOneNot.push_back(edm::Ref<ThingCollection>(parent, 0));
        assert(element.oneNullOneNot.size() == 2);  // we'll check this in our tests
        element.ptr = edm::Ptr<Thing>(&parent->at(i), i);
        assert(element.ptr == edm::refToPtr(element.ref));
        element.ptrVec.push_back(element.ptr);
        element.ptrVec.push_back(edm::Ptr<Thing>(&parent->at(19 - i), 19 - i));
        element.ptrOneNullOneNot.push_back(edm::Ptr<Thing>(nullptr, 0ul));
        element.ptrOneNullOneNot.push_back(edm::Ptr<Thing>(&parent->at(0), 0ul));
        assert(element.ptrOneNullOneNot.size() == 2);  // we'll check this in our tests
        edm::RefProd<ThingCollection> refProd = edm::RefProd<ThingCollection>(parentHandle);
        edm::Ref<ThingCollection> ref = edm::Ref<ThingCollection>(refProd, i);
        element.refToBaseProd = edm::RefToBaseProd<Thing>(refProd);
        element.refToBase = edm::RefToBase<Thing>(ref);
      } else {
        element.a = i;
        element.refProd = edm::RefProd<ThingCollection>(parentHandle);
        element.ref = edm::Ref<ThingCollection>(element.refProd, i);
        element.refVec.push_back(element.ref);
        element.refVec.push_back(element.ref);
        element.refVec.push_back(edm::Ref<ThingCollection>(parentHandle, 19 - i));
        edm::RefVector<ThingCollection>::iterator ri = element.refVec.begin();
        element.refVec.erase(ri);
        element.oneNullOneNot.push_back(edm::Ref<ThingCollection>(parentHandle.id()));
        element.oneNullOneNot.push_back(edm::Ref<ThingCollection>(parentHandle, 0));
        assert(element.oneNullOneNot.size() == 2);  // we'll check this in our tests
        element.ptr = edm::Ptr<Thing>(parentHandle, i);
        assert(element.ptr == edm::refToPtr(element.ref));
        element.ptrVec.push_back(element.ptr);
        element.ptrVec.push_back(edm::Ptr<Thing>(parentHandle, 19 - i));
        element.ptrOneNullOneNot.push_back(edm::Ptr<Thing>(parentHandle.id()));
        element.ptrOneNullOneNot.push_back(edm::Ptr<Thing>(parentHandle, 0));
        assert(element.ptrOneNullOneNot.size() == 2);  // we'll check this in our tests
        edm::RefProd<ThingCollection> refProd = edm::RefProd<ThingCollection>(parentHandle);
        edm::Ref<ThingCollection> ref = edm::Ref<ThingCollection>(refProd, i);
        element.refToBaseProd = edm::RefToBaseProd<Thing>(element.refProd);
        element.refToBase = edm::RefToBase<Thing>(element.ref);
      }
      result.push_back(element);
      //      element.refVec.clear(); // no need to clear; 'element' is created anew on every loop
    }
  }
}  // namespace edmtest
