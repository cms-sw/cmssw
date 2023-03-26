#include "Geometry/EcalTestBeam/interface/EcalTBCrystalMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CLHEP/Random/RandFlat.h"

#include <string>

int main() {
  edm::FileInPath dataFile("Geometry/EcalTestBeam/data/BarrelSM1CrystalCenterElectron120GeV.dat");
  EcalTBCrystalMap theTestMap(dataFile.fullPath());

  long nCrystal = 1700;

  for (int i = 0; i < 10; ++i) {
    int thisCrystal = CLHEP::RandFlat::shootInt(nCrystal);
    double thisEta = 0.;
    double thisPhi = 0.;

    theTestMap.findCrystalAngles(thisCrystal, thisEta, thisPhi);

    std::LogVerbatim("EcalTestBeam") << "Crystal number " << thisCrystal << " eta = " << thisEta << " phi = " << thisPhi;

    int checkThisCrystal = theTestMap.CrystalIndex(thisEta, thisPhi);

    std::LogVerbatim("EcalTestBeam") << "(eta,phi) = " << thisEta << " , " << thisPhi << " corresponds to crystal n. = " << checkThisCrystal;
  }

  return 0;
}
