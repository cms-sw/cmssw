#include "DataFormats/Provenance/interface/Parentage.h"
#include "FWCore/Utilities/interface/Digest.h"
#include <ostream>
#include <sstream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  Parentage::Parentage() : parents_() {
  }

  Parentage::Parentage(std::vector<BranchID> const& parents) :
    parents_(parents) {
  }

  ParentageID
  Parentage::id() const {
    std::ostringstream oss;
    for (auto const& parent : parents_) {
      oss << parent << ' ';
    }
    
    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);
    ParentageID id(md5alg.digest().toString());
    return id;
  }

  void
  Parentage::write(std::ostream&) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
  }
    
  bool
  operator==(Parentage const& a, Parentage const& b) {
    return a.parents() == b.parents();
  }
}
