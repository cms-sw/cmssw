#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>
#include <cassert>
#include <sstream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {


  ProcessConfiguration::ProcessConfiguration() : processName_(), parameterSetID_(), releaseVersion_(), passID_() {}

  ProcessConfiguration::ProcessConfiguration(
                        std::string const& procName,
                        ParameterSetID const& pSetID,
                        ReleaseVersion const& relVersion,
                        PassID const& pass) :
      processName_(procName),
      parameterSetID_(pSetID),
      releaseVersion_(relVersion),
      passID_(pass) { }

  ProcessConfiguration::ProcessConfiguration(
                        std::string const& procName,
                        ReleaseVersion const& relVersion,
                        PassID const& pass) :
      processName_(procName),
      parameterSetID_(),
      releaseVersion_(relVersion),
      passID_(pass) { }

  ParameterSetID const&
  ProcessConfiguration::parameterSetID() const {
    return parameterSetID_;
  }

  ProcessConfigurationID
  ProcessConfiguration::id() const {
    if(pcid().isValid()) {
      return pcid();
    }
    // This implementation is ripe for optimization.
    std::ostringstream oss;
    oss << *this;
    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);
    ProcessConfigurationID tmp(md5alg.digest().toString());
    pcid().swap(tmp);
    return pcid();
  }

  void
  ProcessConfiguration::setParameterSetID(ParameterSetID const& pSetID) {
    assert(parameterSetID_ == ParameterSetID());
    parameterSetID_ = pSetID;
  }

  bool operator<(ProcessConfiguration const& a, ProcessConfiguration const& b) {
    if (a.processName() < b.processName()) return true;
    if (b.processName() < a.processName()) return false;
    if (a.parameterSetID() < b.parameterSetID()) return true;
    if (b.parameterSetID() < a.parameterSetID()) return false;
    if (a.releaseVersion() < b.releaseVersion()) return true;
    if (b.releaseVersion() < a.releaseVersion()) return false;
    if (a.passID() < b.passID()) return true;
    return false;
  }

  std::ostream&
  operator<< (std::ostream& os, ProcessConfiguration const& pc) {
    os << pc.processName() << ' ' 
       << pc.parameterSetID() << ' '
       << pc.releaseVersion() << ' '
       << pc.passID();
    return os;
  }
}
