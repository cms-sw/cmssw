/** \file BowedSurfaceAlignmentDerivatives.cc
 *
 *  $Date: 2010/10/26 20:41:08 $
 *  $Revision: 1.1 $
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Alignment/CommonAlignmentParametrization/interface/KarimakiAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/BowedSurfaceAlignmentDerivatives.h"
#include <math.h>

AlgebraicMatrix 
BowedSurfaceAlignmentDerivatives::operator()(const TrajectoryStateOnSurface &tsos,
					     double uWidth, double vLength,
					     bool doSplit, double ySplit) const
{ 

  AlgebraicMatrix result(N_PARAM, 2);

  // track parameters on surface:
  const AlgebraicVector5 tsosPar(tsos.localParameters().mixedFormatVector());
  // [1] dxdz : direction tangent in local xz-plane
  // [2] dydz : direction tangent in local yz-plane
  // [3] x    : local x-coordinate
  // [4] y    : local y-coordinate
  double myY = tsosPar[4];
  double myLengthV = vLength;
  if (doSplit) { // re-'calibrate' y length and transform myY to be w.r.t. surface middle
    // Some signs depend on whether we are in surface part below or above ySplit:
    const double sign = (tsosPar[4] < ySplit ? +1. : -1.); 
    const double yMiddle = ySplit * 0.5 - sign * vLength * .25; // middle of surface
    myY = tsosPar[4] - yMiddle;
    myLengthV = vLength * 0.5 + sign * ySplit;
  }

  const AlgebraicMatrix karimaki(KarimakiAlignmentDerivatives()(tsos)); // it's just 6x2...
  // copy u, v, w from Karimaki - they are independent of splitting
  result[dx][0] = karimaki[0][0];
  result[dx][1] = karimaki[0][1];
  result[dy][0] = karimaki[1][0];
  result[dy][1] = karimaki[1][1];
  result[dz][0] = karimaki[2][0];
  result[dz][1] = karimaki[2][1];
  const double aScale = gammaScale(uWidth, myLengthV);
  result[drotZ][0] = myY / aScale; // Since karimaki[5][0] == vx;
  result[drotZ][1] = karimaki[5][1] / aScale;

  double uRel = 2. * tsosPar[3] / uWidth;  // relative u (-1 .. +1)
  double vRel = 2. * myY / myLengthV;       // relative v (-1 .. +1)
  // 'range check':
  const double cutOff = 1.5;
  if (uRel < -cutOff) { uRel = -cutOff; } else if (uRel > cutOff) { uRel = cutOff; }
  if (vRel < -cutOff) { vRel = -cutOff; } else if (vRel > cutOff) { vRel = cutOff; }

  // Legendre polynomials renormalized to LPn(1)-LPn(0)=1 (n>0)
  const double uLP0 = 1.0;
  const double uLP1 = uRel;
  const double uLP2 = uRel * uRel - 1./3.;
  const double vLP0 = 1.0;
  const double vLP1 = vRel;
  const double vLP2 = vRel * vRel - 1./3.;

  // 1st order (slopes, replacing angles beta, alpha)
  result[dslopeX][0] = tsosPar[1] * uLP1 * vLP0;
  result[dslopeX][1] = tsosPar[2] * uLP1 * vLP0;
  result[dslopeY][0] = tsosPar[1] * uLP0 * vLP1;
  result[dslopeY][1] = tsosPar[2] * uLP0 * vLP1;
  
  // 2nd order (sagitta)
  result[dsagittaX ][0] = tsosPar[1] * uLP2 * vLP0;
  result[dsagittaX ][1] = tsosPar[2] * uLP2 * vLP0;
  result[dsagittaXY][0] = tsosPar[1] * uLP1 * vLP1;
  result[dsagittaXY][1] = tsosPar[2] * uLP1 * vLP1;
  result[dsagittaY ][0] = tsosPar[1] * uLP0 * vLP2;
  result[dsagittaY ][1] = tsosPar[2] * uLP0 * vLP2;
   
  return result;
}

//------------------------------------------------------------------------------
double BowedSurfaceAlignmentDerivatives::gammaScale(double width, double splitLength)
{
//   return 0.5 * std::sqrt(width*width + splitLength*splitLength);
//   return 0.5 * (std::fabs(width) + std::fabs(splitLength));
  return 0.5 * (width + splitLength);
}
