#ifndef CalibrationIsolatedParticleseCDetIdFromEtaPhi_h
#define CalibrationIsolatedParticleseCDetIdFromEtaPhi_h

#include <cmath>

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

namespace spr{

  // Returns the DetId from eta, phi values in ECAL and HCAL
  const DetId findDetIdECAL( const CaloGeometry* geo, double eta, double phi, bool debug=false) ;
  const DetId findDetIdHCAL( const CaloGeometry* geo, double eta, double phi, bool debug=false) ;
  const DetId findDetIdCalo( const CaloSubdetectorGeometry* geo, double theta, double phi, double radius, bool debug=false) ;

}
#endif
