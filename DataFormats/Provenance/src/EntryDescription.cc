#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  EntryDescription::EntryDescription() :
    parents_(),
    moduleDescriptionID_(),
    moduleDescriptionPtr_()
  { }

  void
  EntryDescription::init() const {
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

  EntryDescriptionID
  EntryDescription::id() const
  {
    // This implementation is ripe for optimization.
    std::ostringstream oss;
    oss << moduleDescriptionID_ << ' ';
    for (std::vector<BranchID>::const_iterator 
	   i = parents_.begin(),
	   e = parents_.end();
	 i != e;
	 ++i)
      {
	oss << *i << ' ';
      }
    
    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);
    return EntryDescriptionID(md5alg.digest().toString());
  }


  void
  EntryDescription::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    os << "Module Description ID = " << moduleDescriptionID() << '\n';
  }
    
  bool
  operator==(EntryDescription const& a, EntryDescription const& b) {
    return
      a.parents() == b.parents()
      && a.moduleDescriptionID() == b.moduleDescriptionID();
  }
}
