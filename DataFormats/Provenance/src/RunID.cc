#include "DataFormats/Provenance/interface/RunID.h"
#include <ostream>

namespace edm {
  std::ostream& operator<<(std::ostream& oStream, RunID const& iID) {
    oStream << "run: " << iID.run();
    return oStream;
  }
}
