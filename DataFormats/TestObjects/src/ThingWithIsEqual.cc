
// $Id$

#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"

namespace edmtest {

  bool ThingWithIsEqual::isProductEqual(ThingWithIsEqual const& newThing) {
    return a == newThing.a;
  }
}
