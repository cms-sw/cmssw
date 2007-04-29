#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

PTrajectoryStateOnDet::PTrajectoryStateOnDet( const LocalTrajectoryParameters& param,
					      float errmatrix[15], unsigned int id,
					      int surfaceSide) :
  theLocalParameters( param), 
  theLocalErrors(15),
  theDetId( id),
  theSurfaceSide( surfaceSide)
{
  //for (int i=0; i<15; i++) theLocalErrors[i] = errmatrix[i]; // not the best way, surely
  theLocalErrors.insert(theLocalErrors.begin(), errmatrix, &errmatrix[15]); // maybe slightly better
}


