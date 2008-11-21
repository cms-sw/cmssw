
// $Id: ThingWithIsEqual.cc,v 1.1 2008/02/04 20:13:15 wdd Exp $

#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"

namespace edmtest {

  bool ThingWithIsEqual::isProductEqual(ThingWithIsEqual const& newThing) const {
    return a == newThing.a;
  }
}
