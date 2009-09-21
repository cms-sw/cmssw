#include "DataFormats/Provenance/interface/MinimalEventID.h"
#include <ostream>
//#include <limits>

namespace edm {

  std::ostream& operator<<(std::ostream& oStream, MinimalEventID const& iID) {
    oStream << "run: " << iID.run() << " event: " << iID.event();
    return oStream;
  }
}
