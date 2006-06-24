#ifndef Common_Provenance_h
#define Common_Provenance_h

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

$Id: Provenance.h,v 1.4 2006/04/26 19:48:36 wmtan Exp $
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

    std::string const& branchName() const {return product.branchName();}
    std::string const& className() const {return product.className();}
    std::string const& moduleLabel() const {return product.moduleLabel();}
    std::string const& moduleName() const {return product.moduleName();}
    PassID passID() const {return product.passID();}
    std::string const& processName() const {return product.processName();}
    ProductID productID() const {return product.productID();}
    std::string const& productInstanceName() const {return product.productInstanceName();}
    std::string const& productType() const {return product.productType();}
    ParameterSetID const& psetID() const {return product.psetID();}
    VersionNumber versionNumber() const {return product.versionNumber();}

    ConditionsID const& conditionsID() const {return event.cid;}
    BranchEntryDescription::CreatorStatus const& creatorStatus() const {return event.status;}
    std::vector<ProductID> const& parents() const {return event.parents;}

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
