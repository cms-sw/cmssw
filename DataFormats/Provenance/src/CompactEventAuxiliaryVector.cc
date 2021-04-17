#include "DataFormats/Provenance/interface/CompactEventAuxiliaryVector.h"
#include <ostream>

namespace edm {
  void CompactEventAuxiliaryVector::CompactEventAuxiliary::write(std::ostream& os) const {
    os << "processGUID = " << processGUID() << std::endl;
    os << id() << " " << time().unixTime() << " " << bunchCrossing() << " " << orbitNumber() << std::endl;
    os << extra_;
  }

  void CompactEventAuxiliaryVector::CompactEventAuxiliaryExtra::write(std::ostream& os) const {
    os << "Process History ID = " << processHistoryID_ << std::endl;
    os << storeNumber_ << " " << isRealData_ << " " << experimentType_ << std::endl;
  }
}  // namespace edm
