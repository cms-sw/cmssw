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
#include <DataFormats/GeometryVector/interface/GlobalPoint.h>

class MagVolume6Faces;
class VolumeBasedMagneticField;

class VolumeGridTester {
public:
  VolumeGridTester(const MagVolume6Faces* vol, const MagProviderInterpol* mp, const VolumeBasedMagneticField* field)
      : volume_(vol), magProvider_(mp), field_(field) {}

  bool testInside() const;
  bool testFind(GlobalPoint gp) const;

private:
  const MagVolume6Faces* volume_;
  const MagProviderInterpol* magProvider_;
  const VolumeBasedMagneticField* field_;

  void dumpProblem(const MFGrid::LocalPoint& lp, double tolerance) const;
};

#endif
