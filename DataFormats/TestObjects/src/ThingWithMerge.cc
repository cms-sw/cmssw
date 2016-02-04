
// $Id: ThingWithMerge.cc,v 1.1 2008/02/04 20:13:15 wdd Exp $

#include "DataFormats/TestObjects/interface/ThingWithMerge.h"

namespace edmtest {

  bool ThingWithMerge::mergeProduct(ThingWithMerge const& newThing) {
    a += newThing.a;
    return true;
  }
}
