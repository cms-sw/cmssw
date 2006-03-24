#ifndef Common_BranchDescription_h
#define Common_BranchDescription_h

/*----------------------------------------------------------------------
  
BranchDescription: The full description of a product and how it came into
existence.

$Id: BranchDescription.h,v 1.2 2006/03/02 05:50:22 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>
#include <string>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Common/interface/ModuleDescription.h"

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
			       boost::shared_ptr<EDProduct const> edp);

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
    mutable boost::shared_ptr<EDProduct const> productPtr_;

    // The branch name, which is currently derivable fron the other
    // attributes.
    mutable std::string branchName_;

    void init() const;

    void write(std::ostream& os) const;

    bool operator<(BranchDescription const& rh) const;

    bool operator==(BranchDescription const& rh) const;

    std::string productType() const {return friendlyClassName_;}
    std::string moduleLabel() const {return module.moduleLabel_;}
    std::string productInstanceName() const {return productInstanceName_;} 
    std::string processName() const {return module.processName_;}
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, const BranchDescription& p) {
    p.write(os);
    return os;
  }
}
#endif
