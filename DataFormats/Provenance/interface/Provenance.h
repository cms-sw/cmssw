#ifndef DataFormats_Provenance_Provenance_h
#define DataFormats_Provenance_Provenance_h

/*----------------------------------------------------------------------

Provenance: The full description of a product and how it came into
existence.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"

#include "boost/shared_ptr.hpp"

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
  class Provenance {
  public:
    Provenance();

    Provenance(boost::shared_ptr<ConstBranchDescription> const& p, ProductID const& pid);

    Parentage const& event() const {return parentage();}
    BranchDescription const& product() const {return branchDescription_->me();}

    BranchDescription const& branchDescription() const {return branchDescription_->me();}
    ConstBranchDescription const& constBranchDescription() const {return *branchDescription_;}
    boost::shared_ptr<ConstBranchDescription> const& constBranchDescriptionPtr() const {return branchDescription_;}

    ProductProvenance* resolve() const;
    ProductProvenance* productProvenance() const {
      if (productProvenanceValid_) return productProvenancePtr_.get();
      return resolve();
    }
    bool productProvenanceValid() const {
      return productProvenanceValid_;
    }
    Parentage const& parentage() const {return productProvenance()->parentage();}
    BranchID const& branchID() const {return product().branchID();}
    std::string const& branchName() const {return product().branchName();}
    std::string const& className() const {return product().className();}
    std::string const& moduleLabel() const {return product().moduleLabel();}
    std::string const& processName() const {return product().processName();}
    std::string const& productInstanceName() const {return product().productInstanceName();}
    std::string const& friendlyClassName() const {return product().friendlyClassName();}
    boost::shared_ptr<BranchMapper> const& store() const {return store_;}
    ProcessHistoryID const& processHistoryID() const {return *processHistoryID_;}
    bool getProcessConfiguration(ProcessConfiguration& pc) const;
    ProcessConfigurationID processConfigurationID() const;
    ParameterSetID psetID() const;
    std::string moduleName() const;
    ReleaseVersion releaseVersion() const;
    std::map<ProcessConfigurationID, ParameterSetID> const& parameterSetIDs() const {
      return product().parameterSetIDs();
    }
    std::map<ProcessConfigurationID, std::string> const& moduleNames() const {
      return product().moduleNames();
    }
    std::set<std::string> const& branchAliases() const {return product().branchAliases();}

    std::vector<BranchID> const& parents() const {return parentage().parents();}

    void write(std::ostream& os) const;

    void setStore(boost::shared_ptr<BranchMapper> store) const {store_ = store;}

    void setProcessHistoryID(ProcessHistoryID const& phid) {processHistoryID_ = &phid;}

    ProductID const& productID() const {return productID_;}

    void setProductProvenance(ProductProvenance const& prov) const;

    void setProductID(ProductID const& pid) {
      productID_ = pid;
    }

    void setBranchDescription(boost::shared_ptr<ConstBranchDescription> const& p) {
      branchDescription_ = p;
    }

    void resetProductProvenance() const;

    void swap(Provenance&);

  private:
    boost::shared_ptr<ConstBranchDescription> branchDescription_;
    ProductID productID_;
    ProcessHistoryID const* processHistoryID_; // Owned by Auxiliary
    mutable bool productProvenanceValid_;
    mutable boost::shared_ptr<ProductProvenance> productProvenancePtr_;
    mutable boost::shared_ptr<BranchMapper> store_;
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
