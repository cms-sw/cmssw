#ifndef FWCore_ServiceRegistry_ProcessContext_h
#define FWCore_ServiceRegistry_ProcessContext_h

/**\class edm::ProcessContext

 Description: Holds pointer to ProcessConfiguration.
 This is intended primarily to be passed to Services
 as an argument to their callback functions.

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

    void setProcessConfiguration(ProcessConfiguration const* processConfiguration);

  private:
    ProcessConfiguration const* processConfiguration_;
  };

  std::ostream& operator<<(std::ostream&, ProcessContext const&);
}  // namespace edm
#endif
