#ifndef Framework_Provenance_h
#define Framework_Provenance_h

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

$Id: Provenance.h,v 1.13 2005/08/02 22:16:39 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>

#include "FWCore/Framework/interface/EventProductDescription.h"
#include "FWCore/EDProduct/interface/ProductID.h"
#include "FWCore/Framework/interface/ProductDescription.h"

/*
  Provenance

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  struct Provenance {
    Provenance();
    explicit Provenance(ProductDescription const& p);

    ~Provenance() {}

    ProductDescription product;
    EventProductDescription event;

    void write(std::ostream& os) const;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, Provenance const& p) {
    p.write(os);
    return os;
  }
}
#endif
