#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>

namespace edm {

  ProcessContext::ProcessContext() :
    processConfiguration_(nullptr),
    parentProcessContext_(nullptr) {
  }

  ProcessContext const&
  ProcessContext::parentProcessContext() const {
    if (!isSubProcess()) {
      throw Exception(errors::LogicError)
        << "ProcessContext::parentProcessContext This function should only be called for SubProcesses.\n"
        << "If necessary, you can check this by calling isSubProcess first.\n";
    }
    return *parentProcessContext_;
  }

  void
  ProcessContext::setProcessConfiguration(ProcessConfiguration const* processConfiguration) {
    processConfiguration_ = processConfiguration;
  }

  void
  ProcessContext::setParentProcessContext(ProcessContext const* parentProcessContext) {
    parentProcessContext_ = parentProcessContext;
  }

  std::ostream& operator<<(std::ostream& os, ProcessContext const& pc) {
    os << "ProcessContext: ";
    if(pc.processConfiguration()) {
      os << *pc.processConfiguration() << "\n";
    } else {
      os << "invalid\n";
      return os;
    }
    if(pc.isSubProcess()) {
      os << "    parent " << pc.parentProcessContext();
    }
    return os;
  }
}
