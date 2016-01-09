#ifndef DataFormats_Provenance_Provenance_h
#define DataFormats_Provenance_Provenance_h

/*----------------------------------------------------------------------

Provenance: The full description of a product and how it came into
existence.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"

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

    BranchDescription const& product() const {return *branchDescription_;}

    BranchDescription const& branchDescription() const {return *branchDescription_;}
    BranchDescription const& constBranchDescription() const {return *branchDescription_;}
    std::shared_ptr<BranchDescription const> const& constBranchDescriptionPtr() const {return branchDescription_;}

    ProductProvenance const* productProvenance() const;
    BranchID const& branchID() const {return product().branchID();}
    std::string const& branchName() const {return product().branchName();}
    std::string const& className() const {return product().className();}
    std::string const& moduleLabel() const {return product().moduleLabel();}
    std::string const& processName() const {return product().processName();}
    std::string const& productInstanceName() const {return product().productInstanceName();}
    std::string const& friendlyClassName() const {return product().friendlyClassName();}
    ProductProvenanceRetriever const* store() const {return store_;}
    ProcessHistory const& processHistory() const {return *processHistory_;}
    bool getProcessConfiguration(ProcessConfiguration& pc) const;
    ReleaseVersion releaseVersion() const;
    std::set<std::string> const& branchAliases() const {return product().branchAliases();}

    void write(std::ostream& os) const;

    void setStore(ProductProvenanceRetriever const* store) {store_ = store;}

    void setProcessHistory(ProcessHistory const& ph) {processHistory_ = &ph;}

    ProductID const& productID() const {return productID_;}

    void setProductID(ProductID const& pid) {
      productID_ = pid;
    }

    void setBranchDescription(std::shared_ptr<BranchDescription const> const& p) {
      branchDescription_ = p;
    }

    void swap(Provenance&);

  private:
    std::shared_ptr<BranchDescription const> branchDescription_;
    ProductID productID_;
    ProcessHistory const* processHistory_; // We don't own this
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
