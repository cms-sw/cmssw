#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include <ostream>

namespace edm {
  std::ostream& operator<<(std::ostream& oStream, LuminosityBlockID const& iID) {
    oStream<< "run: " << iID.run() << " luminosityBlock: " << iID.luminosityBlock();
    return oStream;
  }
}
