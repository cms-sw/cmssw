#ifndef CalibrationIsolatedParticlesDebugInfo_h
#define CalibrationIsolatedParticlesDebugInfo_h

#include <sstream>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"

namespace spr {

  std::ostringstream debugEcalDet(unsigned int, const DetId&);
  void debugEcalDets(unsigned int, std::vector<DetId>&);
  void debugEcalDets(unsigned int, std::vector<DetId>&, std::vector<CaloDirection>&);
  void debugHcalDets(unsigned int, std::vector<DetId>&);
}  // namespace spr

#endif
