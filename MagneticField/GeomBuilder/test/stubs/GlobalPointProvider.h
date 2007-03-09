#ifndef GlobalPointProvider_h
#define GlobalPointProvider_h

/** \class GlobalPointProvider
 *
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include <CLHEP/Random/RandFlat.h>
#include "DataFormats/GeometryVector/interface/Pi.h"

#include <string>



class GlobalPointProvider {
 public:
  GlobalPointProvider(float minR,
		      float maxR,
		      float minPhi,
		      float maxPhi,
		      float minZ,
		      float maxZ) : 
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

    float R = RandFlat::shoot(theMinR,theMaxR);
    float Z = RandFlat::shoot(theMinZ,theMaxZ);
    float phi = RandFlat::shoot(theMinPhi,theMaxPhi);

    GlobalPoint gp(GlobalPoint::Cylindrical(R,phi,Z));

    // if not in barrel, retry
    //    if (barrelOnly && !(theGeometry->inBarrel(gp))) gp=getPoint();

    return gp;
  }


 private:
  float theMinR;
  float theMaxR;  
  float theMinPhi;
  float theMaxPhi;
  float theMinZ;
  float theMaxZ;
};

#endif

