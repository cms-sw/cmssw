#ifndef DataFormats_Provenance_EventEntryDescription_h
#define DataFormats_Provenance_EventEntryDescription_h

/*----------------------------------------------------------------------
  
EventEntryDescription: The event dependent portion of the description of a product
and how it came into existence.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <vector>

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"

/*
  EventEntryDescription

  definitions:
  Product: The WrapperBase to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  class EventEntryDescription {
  public:
    EventEntryDescription();

    ~EventEntryDescription() {}

    // Only the 'salient attributes' are encoded into the ID.
    EntryDescriptionID id() const;

    void write(std::ostream& os) const;

    std::vector<BranchID> const& parents() const { return parents_; }

    void setParents(std::vector<BranchID> const& parents) { parents_ = parents; }

  private:
    // The Branch IDs of the parents
    std::vector<BranchID> parents_;

    Hash<ModuleDescriptionType> moduleDescriptionID_;
  };

  inline std::ostream& operator<<(std::ostream& os, EventEntryDescription const& p) {
    p.write(os);
    return os;
  }

  // Only the 'salient attributes' are testing in equality comparison.
  bool operator==(EventEntryDescription const& a, EventEntryDescription const& b);
  inline bool operator!=(EventEntryDescription const& a, EventEntryDescription const& b) { return !(a == b); }
}  // namespace edm
#endif
