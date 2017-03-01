#ifndef DataFormats_Provenance_ProductProvenance_h
#define DataFormats_Provenance_ProductProvenance_h

/*----------------------------------------------------------------------

ProductProvenance: The event dependent portion of the description of a product
and how it came into existence.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

#include <memory>

#include <iosfwd>
#include <vector>

/*
  ProductProvenance
*/

namespace edm {
  class ProductProvenance {
  public:
    ProductProvenance();
    explicit ProductProvenance(BranchID const& bid);
    ProductProvenance(BranchID const& bid,
                      ParentageID const& id);

    ProductProvenance(BranchID const& bid,
                      std::vector<BranchID> const& parents);

    ProductProvenance(BranchID const& bid,
                      std::vector<BranchID>&& parents);

    ~ProductProvenance() {}

    ProductProvenance makeProductProvenance() const;

    void write(std::ostream& os) const;

    BranchID const& branchID() const {return branchID_;}
    ParentageID const& parentageID() const {return parentageID_;}
    Parentage const& parentage() const;

  private:

    BranchID branchID_;
    ParentageID parentageID_;
  };

  inline
  bool
  operator<(ProductProvenance const& a, ProductProvenance const& b) {
    return a.branchID() < b.branchID();
  }

  inline
  std::ostream&
  operator<<(std::ostream& os, ProductProvenance const& p) {
    p.write(os);
    return os;
  }

  // Only the 'salient attributes' are testing in equality comparison.
  bool operator==(ProductProvenance const& a, ProductProvenance const& b);
  inline bool operator!=(ProductProvenance const& a, ProductProvenance const& b) { return !(a == b); }
  typedef std::vector<ProductProvenance> ProductProvenanceVector;
}
#endif
