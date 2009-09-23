#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include <ostream>

namespace edm {
  bool
  FileFormatVersion::isValid() const {
    return value_ >= 0;
  }

  bool
  FileFormatVersion::lumiNumbers() const {
    return value_ >= 2;
  }

  bool 
  FileFormatVersion::productIDIsInt() const {
    return value_ >= 2;
  }

  bool
  FileFormatVersion::newAuxiliary() const {
    return value_ >= 3;
  }

  bool
  FileFormatVersion::runsAndLumis() const {
    return value_ >= 4;
  }

  bool
  FileFormatVersion::eventHistoryTree() const {
    return value_ >= 7;
  }

  bool
  FileFormatVersion::perEventProductIDs() const {
    return value_ >= 8;
  }

  bool
  FileFormatVersion::splitProductIDs() const {
    return value_ >= 11;
  }

  bool
  FileFormatVersion::fastCopyPossible() const {
    return value_ >= 11;
  }

  bool
  FileFormatVersion::parameterSetsByReference() const {
    return value_ >= 12;
  }

  bool
  FileFormatVersion::triggerPathsTracked() const {
    return value_ >= 13;
  }

  bool
  FileFormatVersion::lumiInEventID() const {
    return value_ >= 14;
  }

  std::ostream&
  operator<< (std::ostream& os, FileFormatVersion const& ff) {
    os << ff.value();
    return os;
  }
}

