#ifndef GlobalPointProvider_h
#define GlobalPointProvider_h

/** \class GlobalPointProvider
 *
 *  No description available.
 *
 *  $Date: 2009/05/25 16:02:09 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include <CLHEP/Random/RandFlat.h>
#include "DataFormats/GeometryVector/interface/Pi.h"

#include <string>



class GlobalPointProvider {
 public:
  GlobalPointProvider(double minR,
		      double maxR,
		      double minPhi,
		      double maxPhi,
		      double minZ,
		      double maxZ) : 
    theMinR(minR),
    theMaxR(maxR),
    theMinPhi(minPhi),
    theMaxPhi(maxPhi),
    theMinZ(minZ),
    theMaxZ(maxZ)
  {}
  
  GlobalPointProvider(bool zSymmetric = true, bool barrelOnly = false) {
    theMinR = 0.;
    theMaxR = 1000.;
    theMinPhi = -Geom::pi();
    theMaxPhi = Geom::pi();
    theMinZ = -1600;
    theMaxZ = 1600;

    if (barrelOnly) {
      theMinZ = -662.;
      theMaxZ = 662.;  
    }
    if (zSymmetric) theMaxZ=0.;
  }
  
  GlobalPoint getPoint() {

    //Turn 
    double R = CLHEP::RandFlat::shoot(theMinR,theMaxR);
    double Z = CLHEP::RandFlat::shoot(theMinZ,theMaxZ);
    double phi = CLHEP::RandFlat::shoot(theMinPhi,theMaxPhi);

    GlobalPoint gp(GlobalPoint::Cylindrical(R,phi,Z));

    // if not in barrel, retry
    //    if (barrelOnly && !(theGeometry->inBarrel(gp))) gp=getPoint();

    return gp;
  }


 private:
  double theMinR;
  double theMaxR;  
  double theMinPhi;
  double theMaxPhi;
  double theMinZ;
  double theMaxZ;
};

#endif

