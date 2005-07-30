#ifndef Framework_ProductDescription_h
#define Framework_ProductDescription_h

/*----------------------------------------------------------------------
  
ProductDescription: The full description of a product and how it came into
existence.

$Id: ProductDescription.h,v 1.7 2005/07/30 04:34:18 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>
#include <string>

#include "FWCore/EDProduct/interface/ProductID.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

/*
  ProductDescription

  definitions:
  The event-independent description of an EDProduct.

*/

namespace edm {
  struct ProductDescription {

    ProductDescription();

    explicit ProductDescription(ModuleDescription const& m,
      std::string const& name, std::string const& fName, std::string const& pin);

    ~ProductDescription() {}

    ModuleDescription module;

    ProductID productID_;

    // the full name of the type of product this is
    std::string fullClassName_;

    // a readable name of the type of product this is
    std::string friendlyClassName_;

    // a user-supplied name to distinguish multiple products of the same type
    // that are produced by the same producer
    std::string productInstanceName_;
    // the last of these is not in the roadmap, but is on the board

    mutable std::string branchName_;

    void init() const;

    void write(std::ostream& os) const;

    bool operator<(ProductDescription const& rh) const;

    bool operator==(ProductDescription const& rh) const;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, const ProductDescription& p) {
    p.write(os);
    return os;
  }
}
#endif
