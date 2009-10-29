#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "FWCore/Utilities/interface/Digest.h"
#include <ostream>
#include <sstream>

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
  }
}
