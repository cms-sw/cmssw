#include "Calibration/IsolatedParticles/interface/DetIdFromEtaPhi.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/Math/interface/Point3D.h"

namespace spr{

  const DetId findDetIdECAL( const CaloGeometry* geo, double eta, double phi) {
    double radius=0;
    int    subdet=0;
    double theta=2.0*std::atan(exp(-eta));
    if (std::abs(eta) > 1.479) {
      radius = 319.2/std::abs(std::cos(theta));
      subdet = EcalEndcap;
    } else {
      radius = 129.4/std::sin(theta);
      subdet = EcalBarrel;
    }
    const CaloSubdetectorGeometry* gECAL = geo->getSubdetectorGeometry(DetId::Ecal,subdet);
    return spr::findDetIdCalo (gECAL, theta, phi, radius);
  }

  const DetId findDetIdHCAL( const CaloGeometry* geo, double eta, double phi) {
    double radius=0;
    double theta=2.0*std::atan(exp(-eta));
    if (std::abs(eta) > 1.392) radius = 402.7/std::abs(std::cos(theta));
    else                       radius = 180.7/std::sin(theta);
    const CaloSubdetectorGeometry* gHCAL = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
    return spr::findDetIdCalo (gHCAL, eta, phi, radius);
  }

  const DetId findDetIdCalo( const CaloSubdetectorGeometry* geo, double theta, double phi, double radius ) {
    
    double rcyl = radius*std::sin(theta);
    double z    = radius*std::cos(theta);
    GlobalPoint  point (rcyl*std::cos(phi),rcyl*std::sin(phi),z);
    const DetId cell = geo->getClosestCell(point);
    return cell;
  }

}
