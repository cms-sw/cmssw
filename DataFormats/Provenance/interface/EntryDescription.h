#ifndef DataFormats_Provenance_EntryDescription_h
#define DataFormats_Provenance_EntryDescription_h

/*----------------------------------------------------------------------
  
EntryDescription: The event dependent portion of the description of a product
and how it came into existence.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <vector>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"

/*
  EntryDescription

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  class EntryDescription {
  public:
    EntryDescription();

    ~EntryDescription() {}

    // Only the 'salient attributes' are encoded into the ID.
    EntryDescriptionID id() const;

    void write(std::ostream& os) const;

    std::vector<ProductID> const& parents() const {return parents_;}
    std::vector<ProductID> & parents() {return parents_;}

    ModuleDescriptionID const& moduleDescriptionID() const {return moduleDescriptionID_;}
    ModuleDescriptionID & moduleDescriptionID() {return moduleDescriptionID_;}


  private:
    // The Branch IDs of the parents
    std::vector<ProductID> parents_;

    // the last of these is not in the roadmap, but is on the board

    ModuleDescriptionID moduleDescriptionID_;

  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, EntryDescription const& p) {
    p.write(os);
    return os;
  }

  // Only the 'salient attributes' are testing in equality comparison.
  bool operator==(EntryDescription const& a, EntryDescription const& b);
  inline bool operator!=(EntryDescription const& a, EntryDescription const& b) { return !(a==b); }
}
#endif
