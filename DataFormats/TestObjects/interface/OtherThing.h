#ifndef TestObjects_OtherThing_h
#define TestObjects_OtherThing_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/TestObjects/interface/ThingCollectionfwd.h"
#include "DataFormats/TestObjects/interface/Thing.h"

namespace edmtest {

  struct OtherThing {
    int                             a;
    edm::RefProd<ThingCollection>   refProd;
    edm::Ref<ThingCollection>       ref;
    edm::RefVector<ThingCollection> refVec;
    edm::RefVector<ThingCollection> oneNullOneNot;
    edm::RefToBase<Thing>           refToBase;
    edm::RefToBaseProd<Thing>       refToBaseProd;
    edm::Ptr<Thing>                 ptr;
    edm::PtrVector<Thing>           ptrVec;
    edm::PtrVector<Thing>           ptrOneNullOneNot;
  };
}

#endif
