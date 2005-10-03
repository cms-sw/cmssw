#ifndef Framework_Provenance_h
#define Framework_Provenance_h

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

$Id: Provenance.h,v 1.15 2005/09/10 03:26:42 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>

#include "FWCore/Framework/interface/BranchEntryDescription.h"
#include "FWCore/EDProduct/interface/ProductID.h"
#include "FWCore/Framework/interface/BranchDescription.h"

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
    explicit Provenance(BranchDescription const& p);

    ~Provenance() {}

    BranchDescription product;
    BranchEntryDescription event;

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
