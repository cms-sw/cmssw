#ifndef VolumeGridTester_h
#define VolumeGridTester_h

/** \class VolumeGridTester
 *
 *  Test the grid for a given volume: each grid point should be
 *  inside the volume.  
 *
 *  \author T. Todorov
 */

#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"

class MagVolume6Faces;

class VolumeGridTester {
public:

  VolumeGridTester( const MagVolume6Faces* vol, const MagProviderInterpol* mp) : 
    volume_(vol), magProvider_(mp) {}

  bool testInside() const;

private:

  const MagVolume6Faces* volume_;
  const MagProviderInterpol* magProvider_;

  void dumpProblem( const MFGrid::LocalPoint& lp, double tolerance) const;

};

#endif
