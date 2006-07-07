#include <sstream>
#include <string>

#include "SealZip/MD5Digest.h"
#include "DataFormats/Common/interface/ProcessConfiguration.h"

/*----------------------------------------------------------------------

$Id: ProcessConfiguration.cc,v 1.2 2006/07/06 18:34:06 wmtan Exp $

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

}
