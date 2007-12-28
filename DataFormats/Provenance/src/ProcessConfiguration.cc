#include <sstream>

#include "FWCore/Utilities/interface/Digest.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: ProcessConfiguration.cc,v 1.2 2007/06/28 23:30:50 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {



  ProcessConfigurationID
  ProcessConfiguration::id() const
  {
    // This implementation is ripe for optimization.
    std::ostringstream oss;
    oss << *this;
    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);
    return ProcessConfigurationID(md5alg.digest().toString());
  }

  std::ostream&
  operator<< (std::ostream& os, ProcessConfiguration const& pc) {
    os << pc.processName_ << ' ' 
       << pc.processGUID_ << ' '
       << pc.parameterSetID_ << ' '
       << pc.releaseVersion_ << ' '
       << pc.passID_;
    return os;
  }
}
