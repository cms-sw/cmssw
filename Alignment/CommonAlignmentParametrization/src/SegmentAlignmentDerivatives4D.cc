/** \file KarimakiAlignmentDerivatives.cc
 *
 *  $Date: 2008/12/12 15:58:07 $
 *  $Revision: 1.1 $
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Alignment/CommonAlignmentParametrization/interface/SegmentAlignmentDerivatives4D.h"


AlgebraicMatrix 
SegmentAlignmentDerivatives4D::operator()( const TrajectoryStateOnSurface &tsos ) const
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

  AlgebraicMatrix aliderivs(6,4);

  //delta_x
  aliderivs[0][0]= -1.0;
  aliderivs[0][1]=  0.0;
  aliderivs[0][2]=  0.0;
  aliderivs[0][3]=  0.0;
  //delta_y
  aliderivs[1][0]=  0.0;
  aliderivs[1][1]= -1.0;
  aliderivs[1][2]=  0.0;
  aliderivs[1][3]=  0.0;
  //delta_z
  aliderivs[2][0]=  tanpsi;
  aliderivs[2][1]=  tantheta;
  aliderivs[2][2]=  tanpsi;
  aliderivs[2][3]=  tantheta;
  //alpha
  aliderivs[3][0]=  vx*tanpsi;
  aliderivs[3][1]=  vx*tantheta;
  aliderivs[3][2]=  0;
  aliderivs[3][3]=  1.0;
  //beta
  aliderivs[4][0]= -ux*tanpsi;   // New beta sign convention
  aliderivs[4][1]= -ux*tantheta; // New beta sign convention
  aliderivs[4][2]= -1.0;  
  aliderivs[4][3]= 0.0; 
  //gamma 
  aliderivs[5][0]=  vx;
  aliderivs[5][1]= -ux;
  aliderivs[5][2]=  tantheta;
  aliderivs[5][3]= -tanpsi;
   
  return(aliderivs);

}
