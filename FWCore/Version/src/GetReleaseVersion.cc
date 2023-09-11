#include "FWCore/Version/interface/GetReleaseVersion.h"



#define STRINGIFY_(x_) #x_
#define STRINGIFY(x_) STRINGIFY_(x_)

namespace edm {
  std::string getReleaseVersion() {
    static std::string const releaseVersion(STRINGIFY(PROJECT_VERSION));
    return releaseVersion; 
  }
}
