#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"
#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"

#include <vector>
#include <iostream>

int main() {
  EcalTBHodoscopeGeometry theTestGeom;

  for (int j = 0; j < theTestGeom.getNPlanes(); ++j) {
    for (int i = 0; i < 1000; ++i) {
      edm::LogVerbatim("EcalGeom") << "Position " << -17. + 34. / 1000. * i << " Plane " << j;
      std::vector<int> firedFibres = theTestGeom.getFiredFibresInPlane(-17. + 34. / 1000. * i, j);
      for (int firedFibre : firedFibres) {
        edm::LogVerbatim("EcalGeom") << firedFibre;

        HodoscopeDetId myDetId = HodoscopeDetId(j, (int)firedFibre);
        edm::LogVerbatim("EcalGeom") << myDetId;
      }
    }
  }

  return 0;
}
