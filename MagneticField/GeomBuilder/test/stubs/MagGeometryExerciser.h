#ifndef MagGeometryExerciser_H
#define MagGeometryExerciser_H

/** \class MagGeometryExerciser
 *  No description available.
 *
 *  $Date: 2005/04/11 17:17:43 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 */

#include <vector>
#include "Geometry/Vector/interface/GlobalPoint.h"

class MagGeometry;
class MagVolume6Faces;


class MagGeometryExerciser {
public:
  /// Constructor
  MagGeometryExerciser(MagGeometry * g);

  /// Destructor
  ~MagGeometryExerciser();


  void testFindVolume(int ntry = 100000); // findVolume(random) test
  void testInside(int ntry = 100000);     // inside(random) test

  void testFieldRandom(int ntry = 1000);// fieldInTesla vs MagneticField::inTesla (random)
  void testFieldVol1();  // fieldInTesla within vol 1 (tiny region)
  void testFieldLinear(int ntry = 1000);// fieldInTesla vs MagneticField::inTesla (track-like pattern)


private:
  // Check if inside succeeds for the given point.
  void testInside(const GlobalPoint & gp);
  // Check if findVolume succeeds for the given point.
  void testFindVolume(const GlobalPoint & gp);

  MagGeometry * theGeometry;
  std::vector<MagVolume6Faces*> volumes;
};
#endif

