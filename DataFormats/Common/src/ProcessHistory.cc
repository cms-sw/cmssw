#include <sstream>
#include <string>
#include "SealZip/MD5Digest.h"

#include "DataFormats/Common/interface/ProcessHistory.h"


namespace edm {
  ProcessHistoryID
  ProcessHistory::id() const
  {
    // This implementation is ripe for optimization.
    // We do not use operator<< because it does not write out everything.
    std::ostringstream oss;
    for (const_iterator i = begin(), e = end(); i != e; ++i) {
      oss << i->processName() << ' '
	  << i->parameterSetID() << ' ' 
	  << i->releaseVersion() << ' '
	  << i->passID() << ' ';
    }
    std::string stringrep = oss.str();
    seal::MD5Digest md5alg;
    md5alg.update(stringrep.data(), stringrep.size());
    return ProcessHistoryID(md5alg.format());
  }
}
