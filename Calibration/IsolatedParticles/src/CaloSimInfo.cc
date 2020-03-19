#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Calibration/IsolatedParticles/interface/CaloConstants.h"
#include "Calibration/IsolatedParticles/interface/CaloSimInfo.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>

//#define EDM_ML_DEBUG

namespace spr {

  double timeOfFlight(DetId id, const CaloGeometry* geo, bool debug) {
    GlobalPoint point = (id.det() == DetId::Hcal)
                            ? (static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(id)))->getPosition(id)
                            : geo->getPosition(id);
    double R = point.mag();
    double tmp = R / CLHEP::c_light / CLHEP::ns;
#ifdef EDM_ML_DEBUG
    if (debug) {
      DetId::Detector det = id.det();
      int subdet = id.subdetId();
      double eta = point.eta();
      double theta = 2.0 * atan(exp(-std::abs(eta)));
      double dist = 0;
      if (det == DetId::Ecal) {
        if (subdet == static_cast<int>(EcalBarrel)) {
          const double rEB = spr::rFrontEB * CLHEP::cm;
          dist = rEB / sin(theta);
        } else if (subdet == static_cast<int>(EcalEndcap)) {
          const double zEE = spr::zFrontEE * CLHEP::cm;
          dist = zEE / cos(theta);
        } else {
          const double zES = spr::zFrontES * CLHEP::cm;
          dist = zES / cos(theta);
        }
      } else if (det == DetId::Hcal) {
        if (subdet == static_cast<int>(HcalBarrel)) {
          const double rHB = spr::rFrontHB * CLHEP::cm;
          dist = rHB / sin(theta);
        } else if (subdet == static_cast<int>(HcalEndcap)) {
          const double zHE = spr::zFrontHE * CLHEP::cm;
          dist = zHE / cos(theta);
        } else if (subdet == static_cast<int>(HcalOuter)) {
          const double rHO = spr::rFrontHO * CLHEP::cm;
          dist = rHO / sin(theta);
        } else {
          const double zHF = spr::zFrontHF * CLHEP::cm;
          dist = zHF / cos(theta);
        }
      }
      double tmp1 = dist / CLHEP::c_light / CLHEP::ns;

      std::cout << "Detector " << det << "/" << subdet << " Eta/Theta " << eta << "/" << theta / CLHEP::deg << " Dist "
                << dist / CLHEP::cm << " R " << R << " TOF " << tmp << ":" << tmp1 << std::endl;
    }
#endif
    return tmp;
  }

}  // namespace spr
