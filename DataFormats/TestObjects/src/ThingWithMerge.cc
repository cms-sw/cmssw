

#include "DataFormats/TestObjects/interface/ThingWithMerge.h"

namespace edmtest {

  bool ThingWithMerge::mergeProduct(ThingWithMerge const& newThing) {
    a += newThing.a;
    return true;
  }
}
