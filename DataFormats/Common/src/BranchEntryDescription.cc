#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "FWCore/Framework/interface/ModuleDescriptionRegistry.h"

/*----------------------------------------------------------------------

$Id: BranchEntryDescription.cc,v 1.1.2.4 2006/06/30 04:30:05 wmtan Exp $

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
    os << "BranchEntryDescription for: " << CreatorStatus();
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
