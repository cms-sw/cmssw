#ifndef Utilities_GetReleaseVersion_h
#define Utilities_GetReleaseVersion_h

#include <string>
#include "FWCore/Utilities/interface/GetEnvironmentVariable.h"

namespace edm {
  inline
  std::string getReleaseVersion () {
    static std::string const releaseVersion(getEnvironmentVariable("CMSSW_VERSION"));
    return releaseVersion; 
  };
}
#endif
