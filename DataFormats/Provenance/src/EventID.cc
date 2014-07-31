#include "DataFormats/Provenance/interface/EventID.h"
#include <ostream>


namespace edm {

  RunNumber_t const EventID::invalidRun = 0U;
  LuminosityBlockNumber_t const EventID::invalidLumi = 0U;
  EventNumber_t const EventID::invalidEvent = 0U;

  std::ostream& operator<<(std::ostream& oStream, EventID const& iID) {
    oStream << "run: " << iID.run() << " lumi: " << iID.luminosityBlock() << " event: " << iID.event();
    return oStream;
  }
}
