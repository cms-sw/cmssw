

#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"

namespace edmtest {

  bool ThingWithIsEqual::isProductEqual(ThingWithIsEqual const& newThing) const {
    return a == newThing.a;
  }
}
