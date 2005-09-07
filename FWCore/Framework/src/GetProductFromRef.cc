#include "FWCore/EDProduct/interface/GetProductFromRef.h"
#include "FWCore/EDProduct/interface/RefBase.h"
#include "FWCore/Framework/interface/EventRegistry.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {
  class EDProduct;

  EDProduct const *getProductFromRef(RefBase const& ref) {
    if (!ref.event()) ref.setEvent(EventRegistry::instance()->getEvent(ref.evtID()));
    return (ref.event()->get(ref.id()).wrapper());
  } 
}

