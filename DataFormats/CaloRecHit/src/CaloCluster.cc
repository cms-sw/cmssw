#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

using namespace reco;

void CaloCluster::setDetector(Detectors theDetector, bool value) {
  
  if(value)
    detectors_ = detectors_ | (1<<theDetector);
  else 
    detectors_ = detectors_ ^ (1<<theDetector);
}



bool CaloCluster::detector(Detectors theDetector) const {

  return (detectors_>>theDetector) & 1;
}

