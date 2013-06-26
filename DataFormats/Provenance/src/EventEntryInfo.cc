#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  EventEntryInfo::EventEntryInfo() :
    branchID_(),
    productID_(),
    entryDescriptionID_()
  {}

  EventEntryInfo::~EventEntryInfo() {}

  void
  EventEntryInfo::write(std::ostream& os) const {
    os << "branch ID = " << branchID() << '\n';
    os << "product ID = " << productID() << '\n';
    os << "entry description ID = " << entryDescriptionID() << '\n';
  }
    
  bool
  operator==(EventEntryInfo const& a, EventEntryInfo const& b) {
    return
      a.branchID() == b.branchID()
      && a.productID() == b.productID()
      && a.entryDescriptionID() == b.entryDescriptionID();
  }
}
