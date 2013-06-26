/** \file BeamSpotAlignmentDerivatives.cc
 *
 *  $Date: 2010/09/10 11:18:29 $
 *  $Revision: 1.1 $
 *  (last update by $Author: mussgill $)
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

#include "Alignment/CommonAlignmentParametrization/interface/BeamSpotAlignmentDerivatives.h"

AlgebraicMatrix 
BeamSpotAlignmentDerivatives::operator()( const TrajectoryStateOnSurface &tsos ) const
{ 
  AlgebraicMatrix aliderivs(4,2);

  if (!tsos.isValid()) return aliderivs;

  // lp.x = transverse impact parameter
  // lp.y = longitudinal impact parameter
  LocalPoint lp = tsos.localPosition();
  double phi = tsos.globalMomentum().phi();
  double dz = lp.y();
  double sinphi = sin(phi);
  double cosphi = cos(phi);

  aliderivs[0][0]=  sinphi;
  aliderivs[0][1]=  0.0;
  aliderivs[1][0]= -cosphi;
  aliderivs[1][1]=  0.0;
  aliderivs[2][0]=  sinphi*dz;
  aliderivs[2][1]=  0.0;
  aliderivs[3][0]= -cosphi*dz;
  aliderivs[3][1]=  0.0;
  
  return(aliderivs);
}
