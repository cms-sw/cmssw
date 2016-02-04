#ifndef MagGeometryExerciser_H
#define MagGeometryExerciser_H

/** \class MagGeometryExerciser
 *  No description available.
 *
 *  $Date: 2008/03/29 14:39:18 $
 *  $Revision: 1.4 $
 *  \author N. Amapane - INFN Torino
 */

#include <vector>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class MagGeometry;
class MagVolume6Faces;


class MagGeometryExerciser {
public:
  /// Constructor
  MagGeometryExerciser(const MagGeometry * g);

  /// Destructor
  ~MagGeometryExerciser();


  void testFindVolume(int ntry = 100000); // findVolume(random) test
  void testInside(int ntry = 100000, float tolerance=0.);     // inside(random) test

  //  void testFieldRandom(int ntry = 1000);// fieldInTesla vs MagneticField::inTesla (random)
  //  void testFieldVol1();  // fieldInTesla within vol 1 (tiny region)
  //  void testFieldLinear(int ntry = 1000);// fieldInTesla vs MagneticField::inTesla (track-like pattern)


private:
  // Check if inside succeeds for the given point.
  bool testInside(const GlobalPoint & gp, float tolerance=0.);
  // Check if findVolume succeeds for the given point.
  bool testFindVolume(const GlobalPoint & gp);

  const MagGeometry * theGeometry;
  std::vector<MagVolume6Faces*> volumes;
};
#endif

