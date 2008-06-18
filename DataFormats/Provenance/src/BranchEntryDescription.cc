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
    moduleDescriptionID_(),
    moduleDescriptionPtr_()
  { }

  BranchEntryDescription::BranchEntryDescription(BranchID const& pid,
	 BranchEntryDescription::CreatorStatus const& status) :
    productID_(pid),
    parents_(),
    cid_(),
    status_(status),
    isPresent_(status == Success),
    moduleDescriptionID_(),
    moduleDescriptionPtr_()
  { }

  void
  BranchEntryDescription::init() const {
    if (moduleDescriptionPtr_.get() == 0) {
      moduleDescriptionPtr_.reset(new ModuleDescription);
      ModuleDescriptionRegistry::instance()->getMapped(moduleDescriptionID_, *moduleDescriptionPtr_);

      // Commented out this assert when implementing merging of run products.
      // When merging run products, it is possible for the ModuleDescriptionIDs to be different,
      // and then the ModuleDescriptionID will be set to invalid.  Then this assert will fail.
      // Queries using the pointer will return values from a default constructed ModuleDescription
      // (empty string and invalid values)
      // bool found = ModuleDescriptionRegistry::instance()->getMapped(moduleDescriptionID_, *moduleDescriptionPtr_);
      // assert(found);
    }
  }

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
      a.productID_ == b.productID_
      && a.creatorStatus() == b.creatorStatus()
      && a.parents() == b.parents()
      && a.moduleDescriptionID() == b.moduleDescriptionID();
  }

  std::auto_ptr<EntryDescription>
  BranchEntryDescription::convertToEntryDescription() const {
    std::auto_ptr<EntryDescription> entryDescription(new EntryDescription);
    entryDescription->parents_ = parents_;
    entryDescription->moduleDescriptionID_ = moduleDescriptionID_;
    entryDescription->moduleDescriptionPtr_ = moduleDescriptionPtr_;
    EntryDescriptionRegistry::instance()->insertMapped(*entryDescription);
    return entryDescription;
  }
}
