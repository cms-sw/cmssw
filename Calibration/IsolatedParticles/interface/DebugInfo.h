#ifndef CalibrationIsolatedParticlesDebugInfo_h
#define CalibrationIsolatedParticlesDebugInfo_h

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"

namespace spr{

  void debugEcalDets(unsigned int, const DetId&, bool);
  void debugEcalDets(unsigned int, std::vector<DetId>&);
  void debugEcalDets(unsigned int, std::vector<DetId>&, 
		     std::vector<CaloDirection>&);
  void debugHcalDets(unsigned int, std::vector<DetId>&);
}

#endif
