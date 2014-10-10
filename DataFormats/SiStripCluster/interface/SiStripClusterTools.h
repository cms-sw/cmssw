#ifndef DataFormatsSiStripClusterSiStripClusterTools_H
#define DataFormatsSiStripClusterSiStripClusterTools_H

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include<numeric>

namespace siStripClusterTools {

  // to be moved and optimized in TrackerCommon when TrackerTopology will support moduleGeometry
  float sensorThicknessInverse (DetId detid)
  {
    if (detid.subdetId()>=SiStripDetId::TIB) {
      SiStripDetId siStripDetId = detid(); 
      if (siStripDetId.subdetId()==SiStripDetId::TOB) return 1.f/0.047f;
      if (siStripDetId.moduleGeometry()==SiStripDetId::W5 || siStripDetId.moduleGeometry()==SiStripDetId::W6 ||
          siStripDetId.moduleGeometry()==SiStripDetId::W7)
	  return 1.f/0.047f;
      return 1.f/0.029f; // so it is TEC ring 1-4 or TIB or TOB;
    } else if (detid.subdetId()==1) return 1.f/0.0285f;
    else return 1.f/0.027f;
  }

  template<typename Iter>
  float chargePerCM(DetId detid, Iter a, Iter b) {
    return float(std::accumulate(a,b,int(0)))*sensorThicknessInverse(detid);
  }

  template<typename Iter>
  float	chargePerCM(DetId detid, Iter a, Iter b, LocalTrajectoryParameters const & tp) {
    return chargePerCM(detid,a,b)*tp.absdz();
  }


}


#endif // DataFormatsSiStripClusterSiStripClusterTools_H

