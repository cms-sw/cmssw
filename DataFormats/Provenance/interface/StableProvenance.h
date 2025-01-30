#ifndef DataFormats_Provenance_StableProvenance_h
#define DataFormats_Provenance_StableProvenance_h

/*----------------------------------------------------------------------

StableProvenance: The full description of a product, excluding the parentage.
The parentage can change from event to event.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"

#include <memory>

#include <iosfwd>
/*
  StableProvenance

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.
*/

namespace edm {
  class StableProvenance {
  public:
    StableProvenance();

    StableProvenance(std::shared_ptr<ProductDescription const> const& p, ProductID const& pid);

    ProductDescription const& productDescription() const { return *productDescription_; }
    std::shared_ptr<ProductDescription const> const& constProductDescriptionPtr() const { return productDescription_; }

    BranchID const& branchID() const { return productDescription().branchID(); }
    BranchID const& originalBranchID() const { return productDescription().originalBranchID(); }
    std::string const& branchName() const { return productDescription().branchName(); }
    std::string const& className() const { return productDescription().className(); }
    std::string const& moduleLabel() const { return productDescription().moduleLabel(); }
    std::string const& processName() const { return productDescription().processName(); }
    std::string const& productInstanceName() const { return productDescription().productInstanceName(); }
    std::string const& friendlyClassName() const { return productDescription().friendlyClassName(); }
    std::set<std::string> const& branchAliases() const { return productDescription().branchAliases(); }

    void write(std::ostream& os) const;

    ProductID const& productID() const { return productID_; }

    void setProductID(ProductID const& pid) { productID_ = pid; }

    void setProductDescription(std::shared_ptr<ProductDescription const> const& p) { productDescription_ = p; }

    void swap(StableProvenance&);

  private:
    std::shared_ptr<ProductDescription const> productDescription_;
    ProductID productID_;
  };

  inline std::ostream& operator<<(std::ostream& os, StableProvenance const& p) {
    p.write(os);
    return os;
  }

  bool operator==(StableProvenance const& a, StableProvenance const& b);

}  // namespace edm
#endif
