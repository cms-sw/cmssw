#ifndef DataFormats_Provenance_Provenance_h
#define DataFormats_Provenance_Provenance_h

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

----------------------------------------------------------------------*/
#include <iosfwd>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"
#include "boost/shared_ptr.hpp"

/*
  Provenance

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  class Provenance {
  public:
    Provenance(ConstBranchDescription const& p, ProductID const& pid);
    Provenance(BranchDescription const& p, ProductID const& pid);
    Provenance(ConstBranchDescription const& p, ProductID const& pid,
	boost::shared_ptr<ProductProvenance> productProvenance);
    Provenance(BranchDescription const& p, ProductID const& pid,
	boost::shared_ptr<ProductProvenance> productProvenance);

    ~Provenance() {}

    Parentage const& event() const {return parentage();}
    BranchDescription const& product() const {return branchDescription_.me();}

    BranchDescription const& branchDescription() const {return branchDescription_.me();}
    ConstBranchDescription const& constBranchDescription() const {return branchDescription_;}
    ProductProvenance const* productProvenancePtr() const {return productProvenancePtr_.get();}
    boost::shared_ptr<ProductProvenance> productProvenanceSharedPtr() const {return productProvenancePtr_;}
    boost::shared_ptr<ProductProvenance> resolve() const;
    ProductProvenance const& productProvenance() const {
      if (productProvenancePtr_.get()) return *productProvenancePtr_;
      return *resolve();
    }
    ProductProvenance& productProvenance() {
      if (productProvenancePtr_.get()) return *productProvenancePtr_;
      return *resolve();
    }
    Parentage const& parentage() const {return productProvenance().parentage();}
    BranchID const& branchID() const {return product().branchID();}
    std::string const& branchName() const {return product().branchName();}
    std::string const& className() const {return product().className();}
    std::string const& moduleLabel() const {return product().moduleLabel();}
    std::string const& processName() const {return product().processName();}
    ProductStatus const& productStatus() const {return productProvenance().productStatus();}
    std::string const& productInstanceName() const {return product().productInstanceName();}
    std::string const& friendlyClassName() const {return product().friendlyClassName();}
    ProcessHistoryID processHistoryID() const {return store_->processHistoryID();}
    ProcessConfigurationID processConfigurationID() const;
    ParameterSetID psetID() const;
    std::string moduleName() const;
    ReleaseVersion const& releaseVersion() const;
    std::map<ProcessConfigurationID, ParameterSetID> const& parameterSetIDs() const {
      return product().parameterSetIDs();
    }
    std::map<ProcessConfigurationID, std::string> const& moduleNames() const {
      return product().moduleNames();
    }
    std::set<std::string> const& branchAliases() const {return product().branchAliases();}

    bool isPresent() const {return productstatus::present(productStatus());}

    std::vector<BranchID> const& parents() const {return parentage().parents();}

    void setProductProvenance(boost::shared_ptr<ProductProvenance> ppr) const;

    void write(std::ostream& os) const;

    void setStore(boost::shared_ptr<BranchMapper> store) const {store_ = store;}

    ProductID const& productID() const {return productID_;}

  private:
    ConstBranchDescription const branchDescription_;
    ProductID productID_;
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
