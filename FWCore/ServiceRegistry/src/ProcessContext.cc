#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>

namespace edm {

  ProcessContext::ProcessContext() : processConfiguration_(nullptr) {}

  void ProcessContext::setProcessConfiguration(ProcessConfiguration const* processConfiguration) {
    processConfiguration_ = processConfiguration;
  }

  std::ostream& operator<<(std::ostream& os, ProcessContext const& pc) {
    os << "ProcessContext: ";
    if (pc.processConfiguration()) {
      os << pc.processConfiguration()->processName() << " " << pc.processConfiguration()->parameterSetID() << "\n";
    } else {
      os << "invalid\n";
      return os;
    }
    return os;
  }
}  // namespace edm
