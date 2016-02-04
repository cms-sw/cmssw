
// $Id: ThingWithIsEqual.cc,v 1.2 2008/11/21 00:00:17 wmtan Exp $

#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"

namespace edmtest {

  bool ThingWithIsEqual::isProductEqual(ThingWithIsEqual const& newThing) const {
    return a == newThing.a;
  }
}
