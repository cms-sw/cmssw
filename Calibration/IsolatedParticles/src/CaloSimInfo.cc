#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Calibration/IsolatedParticles/interface/CaloSimInfo.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include<iostream>

namespace spr{

  double timeOfFlight(DetId id, const CaloGeometry* geo, bool debug) {

    double R   = geo->getPosition(id).mag();
    double tmp = R/CLHEP::c_light/CLHEP::ns;
    if (debug) {
      DetId::Detector det = id.det();
      int subdet   = id.subdetId();
      double eta   = geo->getPosition(id).eta();
      double theta = 2.0*atan(exp(-std::abs(eta)));
      double dist  = 0;
      if (det == DetId::Ecal) {
	if (subdet == static_cast<int>(EcalBarrel)) {
	  const double rEB = 1292*CLHEP::mm;
	  dist = rEB/sin(theta);
	} else if (subdet == static_cast<int>(EcalEndcap)) {
	  const double zEE = 3192*CLHEP::mm;
	  dist = zEE/cos(theta);
	} else {
	  const double zES = 3032*CLHEP::mm;
	  dist = zES/cos(theta);
	}
      } else if (det == DetId::Hcal) {
	if (subdet == static_cast<int>(HcalBarrel)) {
	  const double rHB = 1807*CLHEP::mm;
	  dist = rHB/sin(theta);
	} else if (subdet == static_cast<int>(HcalEndcap)) {
	  const double zHE = 4027*CLHEP::mm;
	  dist = zHE/cos(theta);
	} else if (subdet == static_cast<int>(HcalOuter)) {
	  const double rHO = 3848*CLHEP::mm;
	  dist = rHO/sin(theta);
	} else {
	  const double zHF = 11.15*CLHEP::m;
	  dist = zHF/cos(theta);
	}
      }
      double tmp1 = dist/CLHEP::c_light/CLHEP::ns;

      std::cout << "Detector " << det << "/" << subdet << " Eta/Theta " << eta 
		<< "/" << theta/CLHEP::deg << " Dist " << dist/CLHEP::cm 
		<< " R " << R << " TOF " << tmp << ":" << tmp1 << std::endl;
    }
    return tmp;
  }

}
