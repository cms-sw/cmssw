#include "Calibration/IsolatedParticles/interface/CaloConstants.h"
#include "Calibration/IsolatedParticles/interface/DetIdFromEtaPhi.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

namespace spr{

  const DetId findDetIdECAL( const CaloGeometry* geo, double eta, double phi, bool debug) {
    double radius=0;
    int    subdet=0;
    double theta=2.0*std::atan(exp(-eta));
    if (std::abs(eta) > spr::etaBEEcal) {
      radius = spr::zFrontEE/std::abs(std::cos(theta));
      subdet = EcalEndcap;
    } else {
      radius = spr::rFrontEB/std::sin(theta);
      subdet = EcalBarrel;
    }
    const CaloSubdetectorGeometry* gECAL = geo->getSubdetectorGeometry(DetId::Ecal,subdet);
    if (debug) std::cout << "findDetIdECAL: eta " << eta << " theta " << theta <<" phi " << phi << " radius " << radius << " subdet " << subdet << std::endl;
    return spr::findDetIdCalo (gECAL, theta, phi, radius, debug);
  }

  const DetId findDetIdHCAL( const CaloGeometry* geo, double eta, double phi, bool debug) {
    double radius=0;
    double theta=2.0*std::atan(exp(-eta));
    if (std::abs(eta) > spr::etaBEHcal) 
      radius = spr::zFrontHE/std::abs(std::cos(theta));
    else
      radius = spr::rFrontHB/std::sin(theta);
    const CaloSubdetectorGeometry* gHCAL = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
    if (debug) std::cout << "findDetIdHCAL: eta " << eta <<" theta "<<theta<< " phi " << phi << " radius " << radius << std::endl;
    return spr::findDetIdCalo (gHCAL, theta, phi, radius, debug);
  }

  const DetId findDetIdCalo( const CaloSubdetectorGeometry* geo, double theta, double phi, double radius, bool debug) {
    
    double rcyl = radius*std::sin(theta);
    double z    = radius*std::cos(theta);
    GlobalPoint  point (rcyl*std::cos(phi),rcyl*std::sin(phi),z);
    const DetId cell = geo->getClosestCell(point);
    if (debug) {
      std::cout << "findDetIdCalo: rcyl " << rcyl << " z " << z << " Point " << point << " DetId ";
      if (cell.det() == DetId::Ecal) {
	if (cell.subdetId() == EcalBarrel) std::cout << (EBDetId)(cell);
	else                               std::cout << (EEDetId)(cell);
      } else {
	std::cout << (HcalDetId)(cell);
      }
      std::cout << std::endl;
    }
    return cell;
  }

}
