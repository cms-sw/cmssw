#include "DataFormats/Provenance/interface/Parentage.h"
#include "FWCore/Utilities/interface/Digest.h"
#include <charconv>
#include <sstream>
//#include <cassert>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  Parentage::Parentage() : parents_() {}

  Parentage::Parentage(std::vector<BranchID> const& parents) : parents_(parents) {}

  Parentage::Parentage(std::vector<BranchID>&& parents) : parents_(std::move(parents)) {}

  ParentageID Parentage::id() const {
    //10 is the maximum number of digits for a 2^32 number
    std::array<char, 10 + 1> buf;
    cms::Digest md5alg;
    for (auto const& parent : parents_) {
      //assert(start < end);
      auto res = std::to_chars(buf.data(), buf.data() + buf.size(), parent.id());
      //assert(res.ec == std::errc());
      *res.ptr = ' ';
      md5alg.append(buf.data(), res.ptr - buf.data() + 1);
    }
    ParentageID id(md5alg.digest().bytes);
    return id;
  }

  void Parentage::write(std::ostream&) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
  }

  bool operator==(Parentage const& a, Parentage const& b) { return a.parents() == b.parents(); }
}  // namespace edm
