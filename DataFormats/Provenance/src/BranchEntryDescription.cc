#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  BranchEntryDescription::BranchEntryDescription() :
    productID_(),
    parents_(),
    cid_(),
    status_(Success),
    isPresent_(false),
    moduleDescriptionID_()
  { }

  BranchEntryDescription::BranchEntryDescription(ProductID const& pid,
	 BranchEntryDescription::CreatorStatus const& status) :
    productID_(pid),
    parents_(),
    cid_(),
    status_(status),
    isPresent_(status == Success),
    moduleDescriptionID_()
  { }

  void
  BranchEntryDescription::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    os << "Product ID = " << productID_ << '\n';
    os << "CreatorStatus = " << creatorStatus() << '\n';
    os << "Module Description ID = " << moduleDescriptionID() << '\n';
    os << "Is Present = " << isPresent() << std::endl;
  }
    
  bool
  operator==(BranchEntryDescription const& a, BranchEntryDescription const& b) {
    return
      a.productID() == b.productID()
      && a.creatorStatus() == b.creatorStatus()
      && a.parents() == b.parents()
      && a.moduleDescriptionID() == b.moduleDescriptionID();
  }

  std::auto_ptr<EntryDescription>
  BranchEntryDescription::convertToEntryDescription() const {
    std::auto_ptr<EntryDescription> entryDescription(new EntryDescription);
    entryDescription->parents() = parents();
    entryDescription->moduleDescriptionID() = moduleDescriptionID();
    return entryDescription;
  }
}
