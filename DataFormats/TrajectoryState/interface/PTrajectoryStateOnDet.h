#ifndef PTrajectoryStateOnDet_H
#define PTrajectoryStateOnDet_H

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include<vector>
class Det;

/** Persistent version of a TrajectoryStateOnSurface.
 *  Stores local trajectory parameters and errors and
 *  the id of the Det defining the surface.
 */
class PTrajectoryStateOnDet {
public:

  PTrajectoryStateOnDet() {theLocalErrors.resize(15);}
  virtual ~PTrajectoryStateOnDet() {}

  PTrajectoryStateOnDet( const LocalTrajectoryParameters& param,
			 float errmatrix[15], unsigned int id,
			 int surfaceSide);

  const LocalTrajectoryParameters& parameters() const {return theLocalParameters;}
  const  std::vector<float> errorMatrix() const {return theLocalErrors;}
  const unsigned int detId() const {return theDetId;}
  const int surfaceSide() const    {return theSurfaceSide;}

  virtual PTrajectoryStateOnDet * clone() const {return new PTrajectoryStateOnDet( * this); }

private:

  LocalTrajectoryParameters theLocalParameters;
  std::vector<float> theLocalErrors;
  unsigned int theDetId;
  int          theSurfaceSide;

};

#endif
