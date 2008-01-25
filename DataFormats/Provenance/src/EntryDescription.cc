#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: EntryDescription.cc,v 1.2 2007/05/29 19:15:12 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  EntryDescription::EntryDescription() :
    productID_(),
    parents_(),
    moduleDescriptionID_(),
    moduleDescriptionPtr_()
  { }

  EntryDescription::EntryDescription(ProductID const& pid) :
    productID_(pid),
    parents_(),
    moduleDescriptionID_(),
    moduleDescriptionPtr_()
  { }

  void
  EntryDescription::init() const {
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
  EntryDescription::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    os << "Product ID = " << productID_ << '\n';
    os << "Module Description ID = " << moduleDescriptionID() << '\n';
  }
    
  bool
  operator==(EntryDescription const& a, EntryDescription const& b) {
    return
      a.productID_ == b.productID_
      && a.parents() == b.parents()
      && a.moduleDescriptionID() == b.moduleDescriptionID();
  }

  void
  EntryDescription::mergeEntryDescription(EntryDescription const* entry) {

    assert(productID_ == entry->productID_);

    edm::sort_all(parents_);
    std::vector<ProductID> other = entry->parents_;
    edm::sort_all(other);
    std::vector<ProductID> result;
    std::set_union(parents_.begin(), parents_.end(),
                   other.begin(), other.end(),
                   back_inserter(result)); 
    parents_ = result;
    
    // If they are not equal, set the module description ID to an
    // invalid value.  Currently, there is no way to store multiple
    // values.  An invalid value is better than a partially incorrect
    // value.  In the future, we may need to improve this.
    if (moduleDescriptionID_ != entry->moduleDescriptionID_) {
      moduleDescriptionID_ = ModuleDescriptionID();
    }
  }

}
