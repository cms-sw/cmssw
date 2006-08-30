#ifndef TestObjects_OtherThing_h
#define TestObjects_OtherThing_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TestObjects/interface/ThingCollectionfwd.h"

namespace edmtest {

  struct OtherThing {
    OtherThing():a(), refProd(), ref(), refVec() { }
    ~OtherThing() {}
    int a;
    edm::RefProd<ThingCollection> refProd;
    edm::Ref<ThingCollection> ref;
    edm::RefVector<ThingCollection> refVec;
  };
}

#endif
