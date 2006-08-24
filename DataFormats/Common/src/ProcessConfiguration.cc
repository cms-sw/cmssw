#include <sstream>
#include <string>

#include "SealZip/MD5Digest.h"
#include "DataFormats/Common/interface/ProcessConfiguration.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: ProcessConfiguration.cc,v 1.1 2006/07/07 19:42:35 paterno Exp $

----------------------------------------------------------------------*/

namespace edm {



  ProcessConfigurationID
  ProcessConfiguration::id() const
  {
    // This implementation is ripe for optimization.
    seal::MD5Digest md5alg;
    std::ostringstream oss;
    oss << *this;
    std::string stringrep = oss.str();
    md5alg.update(stringrep.data(), stringrep.size());
    return ProcessConfigurationID(md5alg.format());
  }

  std::ostream&
  operator<< (std::ostream& os, ProcessConfiguration const& pc) {
    os << pc.processName_ << ' ' 
       << pc.parameterSetID_ << ' '
       << pc.releaseVersion_ << ' '
       << pc.passID_;
    return os;
  }
}
