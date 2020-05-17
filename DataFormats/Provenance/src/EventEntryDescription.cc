#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "FWCore/Utilities/interface/Digest.h"
#include <ostream>
#include <sstream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  EventEntryDescription::EventEntryDescription() : parents_() {}

  EntryDescriptionID EventEntryDescription::id() const {
    // This implementation is ripe for optimization.
    std::ostringstream oss;
    oss << moduleDescriptionID_ << ' ';
    for (auto parent : parents_) {
      oss << parent << ' ';
    }

    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);
    return EntryDescriptionID(md5alg.digest().toString());
  }

  void EventEntryDescription::write(std::ostream&) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
  }

  bool operator==(EventEntryDescription const& a, EventEntryDescription const& b) { return a.parents() == b.parents(); }
}  // namespace edm
