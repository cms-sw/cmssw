#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: BranchEntryDescription.cc,v 1.1 2007/03/04 04:48:09 wmtan Exp $

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

  BranchEntryDescription::BranchEntryDescription(ProductID const& pid,
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
      moduleDescriptionPtr_ = boost::shared_ptr<ModuleDescription>(new ModuleDescription);
      bool found = ModuleDescriptionRegistry::instance()->getMapped(moduleDescriptionID_, *moduleDescriptionPtr_);
      assert(found);
    }
  }

  void
  BranchEntryDescription::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    os << "Product ID = " << productID_ << '\n';
    os << "Conditions ID = " << conditionsID() << '\n';
    os << "CreatorStatus = " << creatorStatus() << '\n';
    os << "Module Description ID = " << moduleDescriptionID() << '\n';
    os << "Is Present = " << isPresent() << std::endl;
  }
    
  bool
  operator==(BranchEntryDescription const& a, BranchEntryDescription const& b) {
    return
      a.productID_ == b.productID_
      && a.conditionsID() == b.conditionsID()
      && a.creatorStatus() == b.creatorStatus()
      && a.parents() == b.parents()
      && a.moduleDescriptionID() == b.moduleDescriptionID();
  }
}
