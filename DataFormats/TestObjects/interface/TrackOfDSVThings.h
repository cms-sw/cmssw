#ifndef DataFormats_TestObjects_TrackOfDSVThings_h
#define DataFormats_TestObjects_TrackOfDSVThings_h

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/TestObjects/interface/Thing.h"

namespace edmtest {

  struct TrackOfDSVThings {
    TrackOfDSVThings() = default;

    edm::Ref<edmNew::DetSetVector<Thing>, Thing> ref1;
    edm::Ref<edmNew::DetSetVector<Thing>, Thing> ref2;
    edm::RefVector<edmNew::DetSetVector<Thing>, Thing> refVector1;
  };
}  // namespace edmtest

namespace edmtest {
  typedef std::vector<edmtest::TrackOfDSVThings> TrackOfDSVThingsCollection;
}

#endif
