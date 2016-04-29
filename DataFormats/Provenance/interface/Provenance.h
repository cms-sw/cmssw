#ifndef DataFormats_Provenance_Provenance_h
#define DataFormats_Provenance_Provenance_h

/*----------------------------------------------------------------------

Provenance: The full description of a product and how it came into
existence.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/StableProvenance.h"

#include <memory>

#include <iosfwd>
/*
  Provenance

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  class ProductProvenance;
  class ProductProvenanceRetriever;
  class Provenance {
  public:
    Provenance();

    Provenance(std::shared_ptr<BranchDescription const> const& p, ProductID const& pid);

    Provenance(StableProvenance const&);

    StableProvenance const& stable() const {return stableProvenance_;}
    StableProvenance& stable() {return stableProvenance_;}

    BranchDescription const& branchDescription() const {return stable().branchDescription();}
    std::shared_ptr<BranchDescription const> const& constBranchDescriptionPtr() const {return stable().constBranchDescriptionPtr();}

    ProductProvenance const* productProvenance() const;
    BranchID const& branchID() const {return stable().branchID();}
    std::string const& branchName() const {return stable().branchName();}
    std::string const& className() const {return stable().className();}
    std::string const& moduleLabel() const {return stable().moduleLabel();}
    std::string const& moduleName() const {return stable().moduleName();}
    std::string const& processName() const {return stable().processName();}
    std::string const& productInstanceName() const {return stable().productInstanceName();}
    std::string const& friendlyClassName() const {return stable().friendlyClassName();}
    ProductProvenanceRetriever const* store() const {return store_;}
    ProcessHistory const& processHistory() const {return stable().processHistory();}
    bool getProcessConfiguration(ProcessConfiguration& pc) const {return stable().getProcessConfiguration(pc);}
    ReleaseVersion releaseVersion() const {return stable().releaseVersion();}
    std::set<std::string> const& branchAliases() const {return stable().branchAliases();}

    void write(std::ostream& os) const;

    void setStore(ProductProvenanceRetriever const* store) {store_ = store;}

    void setProcessHistory(ProcessHistory const& ph) {stable().setProcessHistory(ph);}

    ProductID const& productID() const {return stable().productID();}

    void setProductID(ProductID const& pid) {stable().setProductID(pid);}

    void setBranchDescription(std::shared_ptr<BranchDescription const> const& p) {stable().setBranchDescription(p);}

    void swap(Provenance&);

  private:
    StableProvenance stableProvenance_;
    ProductProvenanceRetriever const* store_;
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
