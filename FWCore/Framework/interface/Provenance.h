#ifndef PROVENANCE_HH
#define PROVENANCE_HH

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

$Id: Provenance.h,v 1.12 2005/07/30 23:44:24 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>
#include <vector>

#include "FWCore/Framework/interface/ConditionsID.h"
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
    enum CreatorStatus { Success = 0,
			 ApplicationFailure,
			 InfrastructureFailure,
			 CreatorNotRun };

    Provenance();
    explicit Provenance(ProductDescription const& p);

    ~Provenance() {}

    ProductDescription product;
    ProductID productID_;

    // The EDProduct IDs of the parents
    std::vector<ProductID> parents;

    // a single identifier that describes all the conditions used
    ConditionsID cid; // frame ID?

    // the last of these is not in the roadmap, but is on the board

    // if modules can or will place an object in the event
    // even though something not good occurred, like a timeout, then
    // this may be useful - or if the I/O system makes blank or default
    // constructed objects and we need to distinguish between zero
    // things in a collection between nothing was found and the case
    // where a failure caused nothing to be in the collection.
    // Should a provenance be inserted even if a module fails to 
    // create the output it promised?
    CreatorStatus status;


    void write(std::ostream& os) const;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, const Provenance& p)
  {
    p.write(os);
    return os;
  }
}
#endif
