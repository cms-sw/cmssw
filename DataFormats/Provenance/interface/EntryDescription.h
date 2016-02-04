#ifndef DataFormats_Provenance_EntryDescription_h
#define DataFormats_Provenance_EntryDescription_h

/*----------------------------------------------------------------------
  
EntryDescription: The event dependent portion of the description of a product
and how it came into existence.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <vector>

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"

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

  private:
    // The Branch IDs of the parents
    std::vector<ProductID> parents_;

    // the last of these is not in the roadmap, but is on the board

    Hash<ModuleDescriptionType> moduleDescriptionID_;

  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, EntryDescription const& p) {
    p.write(os);
    return os;
  }
}
#endif
