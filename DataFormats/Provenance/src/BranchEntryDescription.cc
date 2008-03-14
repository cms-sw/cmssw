#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: BranchEntryDescription.cc,v 1.2 2007/05/29 19:15:12 wmtan Exp $

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
  BranchEntryDescription::mergeBranchEntryDescription(BranchEntryDescription const* entry) {

    assert(productID_ == entry->productID_);

    edm::sort_all(parents_);
    std::vector<ProductID> other = entry->parents_;
    edm::sort_all(other);
    std::vector<ProductID> result;
    std::set_union(parents_.begin(), parents_.end(),
                   other.begin(), other.end(),
                   back_inserter(result)); 
    parents_ = result;
    
    assert(cid_ == entry->cid_);

    if (status_ == Success || entry->status_ == Success) status_ = Success;

    if (isPresent_ || entry->isPresent_) isPresent_ = true;

    // If they are not equal, set the module description ID to an
    // invalid value.  Currently, there is no way to store multiple
    // values.  An invalid value is better than a partially incorrect
    // value.  In the future, we may need to improve this.
    if (moduleDescriptionID_ != entry->moduleDescriptionID_) {
      moduleDescriptionID_ = ModuleDescriptionID();
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
