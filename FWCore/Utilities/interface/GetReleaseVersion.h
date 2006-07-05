#ifndef Utilities_GetReleaseVersion_h
#define Utilities_GetReleaseVersion_h

#include <cstdlib>
#include <string>

namespace edm {
  inline
  std::string getReleaseVersion () {
    static std::string const releaseVersion(getenv("CMSSW_VERSION"));
    return releaseVersion; 
  };
}
#endif
