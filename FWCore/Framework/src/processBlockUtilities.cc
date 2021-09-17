#include "FWCore/Framework/interface/processBlockUtilities.h"

#include "FWCore/Framework/interface/Event.h"

namespace edm {

  unsigned int eventProcessBlockIndex(Event const& event, std::string const& processName) {
    return event.processBlockIndex(processName);
  }

}  // namespace edm
