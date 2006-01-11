#ifndef Framework_BranchDescription_h
#define Framework_BranchDescription_h

/*----------------------------------------------------------------------
  
BranchDescription: The full description of a product and how it came into
existence.

$Id: BranchDescription.h,v 1.2 2005/12/28 00:09:58 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>
#include <string>

#include "FWCore/EDProduct/interface/ProductID.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

/*
  BranchDescription

  definitions:
  The event-independent description of an EDProduct.

*/

namespace edm {
  class EDProduct;
  struct BranchDescription {

    BranchDescription();

    explicit BranchDescription(ModuleDescription const& m,
			       std::string const& name, 
			       std::string const& fName, 
			       std::string const& pin, 
			       EDProduct const* edp);

    ~BranchDescription() {}

    ModuleDescription module;

    ProductID productID_;

    // the full name of the type of product this is
    std::string fullClassName_;

    // a readable name of the type of product this is
    std::string friendlyClassName_;

    // a user-supplied name to distinguish multiple products of the same type
    // that are produced by the same producer
    std::string productInstanceName_;

    // A pointer to a default constructed Wrapper<T>, where T is the
    // product type.  If T is a user-defined class, the Wrapper
    // contains a null T*.
    EDProduct const * productPtr_;

    // The branch name, which is currently derivable fron the other
    // attributes.
    mutable std::string branchName_;

    void init() const;

    void write(std::ostream& os) const;

    bool operator<(BranchDescription const& rh) const;

    bool operator==(BranchDescription const& rh) const;

    // It is probably sensible to inline these functions.
    std::string productType() const;
    std::string moduleLabel() const;
    std::string productInstanceName() const;
    std::string processName() const;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, const BranchDescription& p) {
    p.write(os);
    return os;
  }
}
#endif
