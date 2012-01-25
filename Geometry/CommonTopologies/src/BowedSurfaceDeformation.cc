///  \author    : Gero Flucke
///  date       : October 2010
///  $Revision: 1.4 $
///  $Date: 2012/01/25 08:28:39 $
///  (last update by $Author: innocent $)

#include "Geometry/CommonTopologies/interface/BowedSurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformationFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// already included via header:
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
BowedSurfaceDeformation::positionCorrection(const Local2DPoint &localPos,
					    const LocalTrackAngles &localAngles,
					    double length, double width) const
{

// different widthes at high/low y could somehow be treated by theRelWidthLowY
//   if (widthLowY > 0. && widthHighY != widthLowY) {
//     // TEC would always create a warning...
//     edm::LogWarning("UnusableData") << "@SUB=BowedSurfaceDeformation::positionCorrection"
// 				    << "Cannot yet deal with different widthes, take "
// 				    << widthHighY << " not " << widthLowY;
//   }
//   const double width = widthHighY;
  
// try to use vectorization...
  MathVector2D  norm(width,length);
  MathVector2D uvRel = 2*localPos.mathVector()/norm; 
  
  //  double uRel = (width  ? 2. * localPos.x() / width  : 0.);  // relative u (-1 .. +1)
  // double vRel = (length ? 2. * localPos.y() / length : 0.);  // relative v (-1 .. +1)
  // 'range check':
  //const double cutOff = 1.5;
  // if (uvRel.x() < -cutOff) { uvRel.v[0] = -cutOff; } else if (uvRel.x() > cutOff) { uvRel.v[0] = cutOff; }
  // if (uvRel.y() < -cutOff) { uvRel.v[1] = -cutOff; } else if (uvRel.y() > cutOff) { uvRel.v[1] = cutOff; }
  const MathVector2D cutOff(1.5,1.5);
  uvRel = max(uvRel,-cutOff);
  uvRel = min(uvRel,cutOff);
  
  // apply coefficients to Legendre polynomials
  // to get local height relative to 'average'
  const double dw 
    = (uvRel[0] * uvRel[0] - 1./3.) * theSagittaX
    +  uvRel[0] * uvRel[1]          * theSagittaXY
    + (uvRel[1] * uvRel[1] - 1./3.) * theSagittaY;

  
  return Local2DVector(-dw*localAngles);
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
