#include "FWCore/Framework/src/ESGlobalMutex.h"

namespace edm {

  static std::recursive_mutex s_esGlobalMutex;

  std::recursive_mutex& esGlobalMutex() { return s_esGlobalMutex; }

}  // namespace edm
