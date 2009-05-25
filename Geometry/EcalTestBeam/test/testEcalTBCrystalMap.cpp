#include "Geometry/EcalTestBeam/interface/EcalTBCrystalMap.h"

#include "CLHEP/Random/RandFlat.h"

#include <string>


using namespace std;

int main() {

  const string MapFileName = "BarrelSM1CrystalCenterElectron120GeV.dat";

  EcalTBCrystalMap theTestMap(MapFileName);

  long nCrystal = 1700;

  for ( int i = 0 ; i < 10 ; ++i ) {

    int thisCrystal = CLHEP::RandFlat::shootInt(nCrystal);
    double thisEta = 0.;
    double thisPhi = 0.;

    theTestMap.findCrystalAngles(thisCrystal, thisEta, thisPhi);

    cout << "Crystal number " << thisCrystal << " eta = " << thisEta << " phi = " << thisPhi << endl;

    int checkThisCrystal = theTestMap.CrystalIndex(thisEta, thisPhi);

    cout << "(eta,phi) = " << thisEta << " , " << thisPhi << " corresponds to crystal n. = " << checkThisCrystal << endl;

  }

  return 0;

}
