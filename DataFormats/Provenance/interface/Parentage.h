#ifndef DataFormats_Provenance_Parentage_h
#define DataFormats_Provenance_Parentage_h

/*----------------------------------------------------------------------
  
Parentage: The products that were read in producing this product.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ParentageID.h"

#include <iosfwd>
#include <vector>

/*
  Parentage

  definitions:
  Product: The WrapperBase to which a provenance object is associated

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  class Parentage {
  public:
    Parentage();

    explicit Parentage(std::vector<BranchID> const& parents);
    explicit Parentage(std::vector<BranchID>&& parents);

    Parentage(Parentage const&) = default;
    Parentage(Parentage&&) = default;

    Parentage& operator=(Parentage const&) = default;
    Parentage& operator=(Parentage&&) = default;

    ~Parentage() {}

    ParentageID id() const;

    void write(std::ostream& os) const;

    std::vector<BranchID> const& parents() const { return parents_; }
    std::vector<BranchID>& parentsForUpdate() { return parents_; }
    void setParents(std::vector<BranchID> parents) { parents_ = std::move(parents); }
    void swap(Parentage& other) { parents_.swap(other.parents_); }
    void initializeTransients() { transient_.reset(); }
    struct Transients {
      Transients() {}
      void reset() {}
    };

  private:
    // The Branch IDs of the parents
    std::vector<BranchID> parents_;
    Transients transient_;
  };

  // Free swap function
  inline void swap(Parentage& a, Parentage& b) { a.swap(b); }

  inline std::ostream& operator<<(std::ostream& os, Parentage const& p) {
    p.write(os);
    return os;
  }

  // Only the 'salient attributes' are testing in equality comparison.
  bool operator==(Parentage const& a, Parentage const& b);
  inline bool operator!=(Parentage const& a, Parentage const& b) { return !(a == b); }
}  // namespace edm
#endif
