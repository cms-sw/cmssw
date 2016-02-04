#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  BranchEntryDescription::BranchEntryDescription() :
    productID_(),
    parents_(),
    cid_(),
    status_(Success),
    isPresent_(false)
  { }

  BranchEntryDescription::~BranchEntryDescription() {}
}
