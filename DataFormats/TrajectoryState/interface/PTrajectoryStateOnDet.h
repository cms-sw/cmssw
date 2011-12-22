#ifndef PTrajectoryStateOnDet_H
#define PTrajectoryStateOnDet_H

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

/** Persistent version of a TrajectoryStateOnSurface.
 *  Stores local trajectory parameters and errors and
 *  the id of the Det defining the surface.
 */
class PTrajectoryStateOnDet {
public:

  PTrajectoryStateOnDet() {}

  PTrajectoryStateOnDet( const LocalTrajectoryParameters& param,
			 unsigned int id,
			 int surfaceSide) :   
    theLocalParameters( param), 
    theDetId( id),
    theSurfaceSide( surfaceSide) {theLocalErrors[0]=-99999.e10; }

  PTrajectoryStateOnDet( const LocalTrajectoryParameters& param,
			 float errmatrix[15], unsigned int id,
			 int surfaceSide) :   
    theLocalParameters( param), 
    theDetId( id),
    theSurfaceSide( surfaceSide)
  {
    for (int i=0; i<15; i++) theLocalErrors[i] = errmatrix[i]; // let's try this way
  }


  const LocalTrajectoryParameters& parameters() const {return theLocalParameters;}
  bool hasError() const { return theLocalErrors[0] > -1.e10; }
  float & error(int i)  {return theLocalErrors[i];}
  float   error(int i) const {return theLocalErrors[i];}
  const unsigned int detId() const {return theDetId;}
  const int surfaceSide() const    {return theSurfaceSide;}

  // virtual PTrajectoryStateOnDet * clone() const {return new PTrajectoryStateOnDet( * this); }

private:

  LocalTrajectoryParameters theLocalParameters;
  float theLocalErrors[15];
  unsigned int theDetId;
  int          theSurfaceSide;

};

#endif
