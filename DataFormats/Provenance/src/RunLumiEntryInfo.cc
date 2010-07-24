#include "DataFormats/Provenance/interface/RunLumiEntryInfo.h"
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  RunLumiEntryInfo::RunLumiEntryInfo() :
    branchID_(),
    productStatus_(productstatus::uninitialized())
  {}

  RunLumiEntryInfo::~RunLumiEntryInfo() {}

  void
  RunLumiEntryInfo::write(std::ostream& os) const {
    os << "branch ID = " << branchID() << '\n';
    os << "product status = " << productStatus() << '\n';
  }
    
  bool
  operator==(RunLumiEntryInfo const& a, RunLumiEntryInfo const& b) {
    return
      a.branchID() == b.branchID()
      && a.productStatus() == b.productStatus();
  }
}
