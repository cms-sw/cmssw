#ifndef Framework_EventProductDescription_h
#define Framework_EventProductDescription_h

/*----------------------------------------------------------------------
  
EventProductDescription: The event dependent portion of the description of a product
and how it came into existence.

$Id: EventProductDescription.h,v 1.13 2005/08/02 22:16:39 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>
#include <vector>

#include "FWCore/Framework/interface/ConditionsID.h"
#include "FWCore/EDProduct/interface/ProductID.h"

/*
  EventProductDescription

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  struct EventProductDescription {
    enum CreatorStatus { Success = 0,
			 ApplicationFailure,
			 InfrastructureFailure,
			 CreatorNotRun };

    EventProductDescription ();

    ~EventProductDescription() {}

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

    bool operator==(EventProductDescription const& rh) const;

    void write(std::ostream& os) const;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, EventProductDescription const& p) {
    p.write(os);
    return os;
  }
}
#endif
