///  \author    : Gero Flucke
///  date       : October 2010
///  $Revision$
///  $Date$
///  (last update by $Author$)

#include "Geometry/CommonTopologies/interface/BowedSurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformationFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// already included via header:
// #include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
// #include <vector>

//------------------------------------------------------------------------------
BowedSurfaceDeformation::BowedSurfaceDeformation(const std::vector<double> &pars)
  : theSagittaX (pars.size() > 0 ? pars[0] : 0.),
    theSagittaXY(pars.size() > 1 ? pars[1] : 0.),
    theSagittaY (pars.size() > 2 ? pars[2] : 0.)
{
  if (pars.size() != minParameterSize()) {
    edm::LogError("BadSetup") << "@SUB=BowedSurfaceDeformation"
                              << "Input vector of wrong size " << pars.size()
                              << " instead of " << minParameterSize() << ", filled up with zeros!";
  }
}

//------------------------------------------------------------------------------
BowedSurfaceDeformation* BowedSurfaceDeformation::clone() const
{
  return new BowedSurfaceDeformation(theSagittaX, theSagittaXY, theSagittaY);
}

//------------------------------------------------------------------------------
int BowedSurfaceDeformation::type() const
{
  return SurfaceDeformationFactory::kBowedSurface;
}

//------------------------------------------------------------------------------
SurfaceDeformation::Local2DVector 
BowedSurfaceDeformation::positionCorrection(const AlgebraicVector5 &trackPred,
					    double length, double width) const
{
  // AlgebraicVector5 &trackPred:
  // [1] dxdz : direction tangent in local xz-plane
  // [2] dydz : direction tangent in local yz-plane
  // [3] x    : local x-coordinate
  // [4] y    : local y-coordinate

// different widthes at high/low y could somehow be treated by theRelWidthLowY
//   if (widthLowY > 0. && widthHighY != widthLowY) {
//     // TEC would always create a warning...
//     edm::LogWarning("UnusableData") << "@SUB=BowedSurfaceDeformation::positionCorrection"
// 				    << "Cannot yet deal with different widthes, take "
// 				    << widthHighY << " not " << widthLowY;
//   }
//   const double width = widthHighY;
  
  double uRel = (width  ? 2. * trackPred[3] / width  : 0.);  // relative u (-1 .. +1)
  double vRel = (length ? 2. * trackPred[4] / length : 0.);  // relative v (-1 .. +1)
  // 'range check':
  const double cutOff = 1.5;
  if (uRel < -cutOff) { uRel = -cutOff; } else if (uRel > cutOff) { uRel = cutOff; }
  if (vRel < -cutOff) { vRel = -cutOff; } else if (vRel > cutOff) { vRel = cutOff; }
  
  // apply coefficients to Legendre polynomials
  // to get local height relative to 'average'
  const double dw 
    = (uRel * uRel - 1./3.) * theSagittaX
    +  uRel * vRel          * theSagittaXY
    + (vRel * vRel - 1./3.) * theSagittaY;

  // positive dxdz/dydz and positive dw mean negative shift in x/y: 
  const Local2DVector::ScalarType x = -dw * trackPred[1]; // [1] = dxdz
  const Local2DVector::ScalarType y = -dw * trackPred[2]; // [2] = dydz
  
  return Local2DVector(x, y);
}

//------------------------------------------------------------------------------
bool BowedSurfaceDeformation::add(const SurfaceDeformation &other)
{
  if (other.type() == this->type()) {
    const std::vector<double> otherParams(other.parameters());
    if (otherParams.size() == 3) { // double check!
      theSagittaX  += otherParams[0]; // bows can simply be added up
      theSagittaXY += otherParams[1];
      theSagittaY  += otherParams[2];

      return true;
    }
  }

  return false;
}
  
//------------------------------------------------------------------------------
std::vector<double> BowedSurfaceDeformation::parameters() const
{
  std::vector<double> result(3);
  result[0] = theSagittaX;
  result[1] = theSagittaXY;
  result[2] = theSagittaY;

  return result;
}
