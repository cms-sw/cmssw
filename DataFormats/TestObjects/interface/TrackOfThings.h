#ifndef DataFormats_TestObjects_TrackOfThings_h
#define DataFormats_TestObjects_TrackOfThings_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"

namespace edmtest {

  struct TrackOfThings {
    ~TrackOfThings() { }
    TrackOfThings() { }

    edm::Ref<ThingCollection> ref1;
    edm::Ref<ThingCollection> ref2;
    edm::RefVector<ThingCollection> refVector1;

    edm::Ptr<Thing> ptr1;
    edm::Ptr<Thing> ptr2;
    edm::PtrVector<Thing> ptrVector1;

    edm::RefToBase<Thing> refToBase1;
    edm::RefToBaseVector<Thing> refToBaseVector1;
  };
}

namespace edmtest {
  typedef std::vector<edmtest::TrackOfThings> TrackOfThingsCollection;
}

#endif
