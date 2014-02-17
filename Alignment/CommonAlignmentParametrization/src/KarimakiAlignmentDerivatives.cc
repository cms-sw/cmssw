/** \file KarimakiAlignmentDerivatives.cc
 *
 *  $Date: 2007/05/02 21:01:53 $
 *  $Revision: 1.7 $
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Alignment/CommonAlignmentParametrization/interface/KarimakiAlignmentDerivatives.h"


AlgebraicMatrix 
KarimakiAlignmentDerivatives::operator()( const TrajectoryStateOnSurface &tsos ) const
{ 

  // Get track parameters on surface
  AlgebraicVector5 alivec = tsos.localParameters().mixedFormatVector();

  // [0] q/p  : charged: charge (+ or - one) divided by magnitude of momentum
  //            neutral : inverse magnitude of momentum
  // [1] dxdz : direction tangent in local xz-plane
  // [2] dydz : direction tangent in local yz-plane
  // [3] x    : local x-coordinate
  // [4] y    : local y-coordinate

  double tanpsi   = alivec[1];
  double tantheta = alivec[2];
  double ux       = alivec[3];
  double vx       = alivec[4];

  AlgebraicMatrix aliderivs(6,2);

  aliderivs[0][0]= -1.0;
  aliderivs[0][1]=  0.0;
  aliderivs[1][0]=  0.0;
  aliderivs[1][1]= -1.0;
  aliderivs[2][0]=  tanpsi;
  aliderivs[2][1]=  tantheta;
  aliderivs[3][0]=  vx*tanpsi;
  aliderivs[3][1]=  vx*tantheta;
  aliderivs[4][0]= -ux*tanpsi;   // New beta sign convention
  aliderivs[4][1]= -ux*tantheta; // New beta sign convention
  aliderivs[5][0]=  vx;
  aliderivs[5][1]= -ux;
   
  return(aliderivs);

}
