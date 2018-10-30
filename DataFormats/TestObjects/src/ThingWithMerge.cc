

#include "DataFormats/TestObjects/interface/ThingWithMerge.h"

#include <utility>

namespace edmtest {

  bool ThingWithMerge::mergeProduct(ThingWithMerge const& newThing) {
    a += newThing.a;
    return true;
  }

  void ThingWithMerge::swap(ThingWithMerge& iOther) {
    std::swap(a, iOther.a);
  }
}
