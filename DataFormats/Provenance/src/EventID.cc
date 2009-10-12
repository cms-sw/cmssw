#include "DataFormats/Provenance/interface/EventID.h"
#include <ostream>

namespace edm {
  std::ostream& operator<<(std::ostream& oStream, EventID const& iID) {
    oStream << "run: " << iID.run() << " lumi: " << iID.luminosityBlock() << " event: " << iID.event();
    return oStream;
  }
}
