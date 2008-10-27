#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  EntryDescription::EntryDescription() :
    parents_(),
    moduleDescriptionID_()
  { }

  EntryDescriptionID
  EntryDescription::id() const
  {
    // This implementation is ripe for optimization.
    std::ostringstream oss;
    oss << moduleDescriptionID_ << ' ';
    for (std::vector<ProductID>::const_iterator 
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
