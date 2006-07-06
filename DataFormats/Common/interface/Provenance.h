#ifndef Common_Provenance_h
#define Common_Provenance_h

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

$Id: Provenance.h,v 1.5.2.9 2006/07/04 13:56:44 wmtan Exp $
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
    std::string const& moduleName() const {return event.moduleName();}
    PassID const& passID() const {return event.passID();}
    std::string const& processName() const {return product.processName();}
    ProductID const& productID() const {return product.productID();}
    std::string const& productInstanceName() const {return product.productInstanceName();}
    std::string const& friendlyClassName() const {return product.friendlyClassName();}
    std::set<ParameterSetID> const& psetIDs() const {return product.psetIDs();}
    ParameterSetID const& psetID() const {return event.psetID();}
    ReleaseVersion const& releaseVersion() const {return event.releaseVersion();}
    std::set<std::string> const& branchAliases() const {return product.branchAliases();}
    ModuleDescriptionID const& moduleDescriptionID() const {return event.moduleDescriptionID();}
    ModuleDescription const& moduleDescription() const {return event.moduleDescription();}

    ConditionsID const& conditionsID() const {return event.conditionsID();}
    BranchEntryDescription::CreatorStatus const& creatorStatus() const {return event.creatorStatus();}
    bool const& isPresent() const {return event.isPresent();}
    std::vector<ProductID> const& parents() const {return event.parents();}

    void write(std::ostream& os) const;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, Provenance const& p) {
    p.write(os);
    return os;
  }

  bool operator==(Provenance const& a, Provenance const& b);

}
#endif
