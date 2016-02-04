#include "DataFormats/Provenance/interface/Parentage.h"
#include "FWCore/Utilities/interface/Digest.h"
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
  Parentage::id() const {
    // This implementation is ripe for optimization.
    if(parentageID().isValid()) {
      return parentageID();
    }
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
    ParentageID tmp(md5alg.digest().toString());
    parentageID().swap(tmp);
    return parentageID();
  }

  void
  Parentage::write(std::ostream&) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
  }
    
  bool
  operator==(Parentage const& a, Parentage const& b) {
    return
      a.parents() == b.parents();
  }
}
