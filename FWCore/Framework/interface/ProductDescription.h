#ifndef Framework_ProductDescription_h
#define Framework_ProductDescription_h

/*----------------------------------------------------------------------
  
ProductDescription: The full description of a product and how it came into
existence.

$Id: ProductDescription.h,v 1.2 2005/07/22 23:48:14 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>
#include <string>

#include "FWCore/EDProduct/interface/ProductID.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/BranchKey.h"

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

    ProductID product_id;

    // the full name of the type of product this is
    std::string full_product_type_name;

    // a readable name of the type of product this is
    std::string friendly_product_type_name;

    // a user-supplied name to distinguish multiple products of the same type
    // that are produced by the same producer
    std::string product_instance_name;
    // the last of these is not in the roadmap, but is on the board

    mutable BranchKey branchKey;

    void init() const;

    void write(std::ostream& os) const;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, const ProductDescription& p) {
    p.write(os);
    return os;
  }

  inline
  bool
  operator==(const ProductDescription& a, const ProductDescription& b) {
    return
      a.module == b.module 
      && a.full_product_type_name == b.full_product_type_name
      && a.friendly_product_type_name == b.friendly_product_type_name
      && a.product_instance_name == b.product_instance_name;
  }
}
#endif
