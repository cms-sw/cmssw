#ifndef Framework_Provenance_h
#define Framework_Provenance_h

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

$Id: Provenance.h,v 1.16 2005/10/03 19:03:01 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>

#include "FWCore/Framework/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/ProductID.h"
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
