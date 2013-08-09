#ifndef FWCore_ServiceRegistry_ProcessContext_h
#define FWCore_ServiceRegistry_ProcessContext_h

/**\class edm::ProcessContext

 Description: Holds pointer to ProcessConfiguration and
if this is a SubProcess also a pointer to the parent
ProcessContext. This is intended primarily to be passed
to Services as an argument to their callback functions.

 Usage:


*/
//
// Original Author: W. David Dagenhart
//         Created: 7/2/2013

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"

#include <iosfwd>
#include <string>

namespace edm {

  class ProcessContext {

  public:

    ProcessContext();

    std::string const& processName() const { return processConfiguration_->processName(); }
    ParameterSetID const& parameterSetID() const { return processConfiguration_->parameterSetID(); }
    ProcessConfiguration const* processConfiguration() const { return processConfiguration_; }
    bool isSubProcess() const { return parentProcessContext_ != nullptr; }
    ProcessContext const& parentProcessContext() const;

    void setProcessConfiguration(ProcessConfiguration const* processConfiguration);
    void setParentProcessContext(ProcessContext const* parentProcessContext);

  private:

    ProcessConfiguration const* processConfiguration_;

    // If this is a SubProcess this points to the parent process,
    // otherwise it is null.
    ProcessContext const* parentProcessContext_;
  };

  std::ostream& operator<<(std::ostream&, ProcessContext const&);
}
#endif
