#ifndef PTrajectoryStateOnDet_H
#define PTrajectoryStateOnDet_H

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

class Det;

/** Persistent version of a TrajectoryStateOnSurface.
 *  Stores local trajectory parameters and errors and
 *  the id of the Det defining the surface.
 */
class PTrajectoryStateOnDet {
public:

  PTrajectoryStateOnDet() {}

  PTrajectoryStateOnDet( const LocalTrajectoryParameters& param,
			 float errmatrix[15], unsigned int id,
			 int surfaceSide);

  const LocalTrajectoryParameters& parameters() const {return theLocalParameters;}
  const float* errorMatrix() const {return theLocalErrors;}
  const unsigned int detId() const {return theDetId;}
  const int surfaceSide() const    {return theSurfaceSide;}

private:

  LocalTrajectoryParameters theLocalParameters;
  float theLocalErrors[15];
  unsigned int theDetId;
  int          theSurfaceSide;

};

#endif
