#include "Calibration/IsolatedParticles/interface/DebugInfo.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

namespace spr {

  std::ostringstream debugEcalDet(unsigned int i, const DetId& det) {
    std::ostringstream st1;
    st1 << "Cell [" << i << "] 0x";
    if (det.subdetId() == EcalBarrel) {
      EBDetId id(det);
      st1 << std::hex << det() << std::dec << " " << id;
    } else if (det.subdetId() == EcalEndcap) {
      EEDetId id(det);
      st1 << std::hex << det() << std::dec << " " << id;
    } else {
      st1 << std::hex << det() << std::dec << " Unknown Type";
    }
    return st1;
  }

  void debugEcalDets(unsigned int last, std::vector<DetId>& vdets) {
    for (unsigned int i = last; i < vdets.size(); ++i) {
      edm::LogVerbatim("IsoTrack") << spr::debugEcalDet(i, vdets[i]).str();
    }
  }

  void debugEcalDets(unsigned int last, std::vector<DetId>& vdets, std::vector<CaloDirection>& dirs) {
    for (unsigned int i = last; i < vdets.size(); ++i) {
      edm::LogVerbatim("IsoTrack") << spr::debugEcalDet(i, vdets[i]).str() << " along " << dirs[i] << std::endl;
    }
  }

  void debugHcalDets(unsigned int last, std::vector<DetId>& vdets) {
    for (unsigned int i = last; i < vdets.size(); ++i) {
      HcalDetId id = vdets[i]();
      edm::LogVerbatim("IsoTrack") << "Cell [" << i << "] 0x" << std::hex << vdets[i]() << std::dec << " " << id;
    }
  }
}  // namespace spr
