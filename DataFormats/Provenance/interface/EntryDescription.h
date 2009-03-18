#ifndef DataFormats_Provenance_EntryDescription_h
#define DataFormats_Provenance_EntryDescription_h

/*----------------------------------------------------------------------
  
EntryDescription: The event dependent portion of the description of a product
and how it came into existence.

Obsolete

----------------------------------------------------------------------*/
#include <vector>

#include "DataFormats/Provenance/interface/ProductID.h"

/*
  EntryDescription

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  class EntryDescription {
  public:
    EntryDescription();

    ~EntryDescription();

    std::vector<ProductID> const& parents() const {return parents_;}
    std::vector<ProductID> & parents() {return parents_;}

  private:
    // The Branch IDs of the parents
    std::vector<ProductID> parents_;

  };
}
#endif
