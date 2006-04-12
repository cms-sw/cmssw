#ifndef Common_Provenance_h
#define Common_Provenance_h

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

$Id: Provenance.h,v 1.1 2006/02/08 00:44:23 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>

#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Common/interface/BranchDescription.h"

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

    std::string branchName() const {return product.branchName();}
    std::string className() const {return product.className();}
    std::string moduleLabel() const {return product.moduleLabel();}
    std::string moduleName() const {return product.moduleName();}
    std::string processName() const {return product.processName();}
    std::string productInstanceName() const {return product.productInstanceName();}
    std::string productType() const {return product.productType();}

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
