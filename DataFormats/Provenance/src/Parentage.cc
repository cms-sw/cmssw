#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include <ostream>
#include <sstream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  Parentage::Parentage() :
    parents_()
  {}

  Parentage::Parentage(std::vector<BranchID> const& parents) :
    parents_(parents)
  {}

  ParentageID
  Parentage::id() const
  {
    // This implementation is ripe for optimization.
    std::ostringstream oss;
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
    return ParentageID(md5alg.digest().toString());
  }


  void
  Parentage::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
  }
    
  bool
  operator==(Parentage const& a, Parentage const& b) {
    return
      a.parents() == b.parents();
  }
}
