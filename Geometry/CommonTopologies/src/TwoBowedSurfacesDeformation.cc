///  \author    : Gero Flucke
///  date       : October 2010
///  $Revision: 1.1 $
///  $Date: 2010/10/26 19:00:00 $
///  (last update by $Author: flucke $)

#include "Geometry/CommonTopologies/interface/TwoBowedSurfacesDeformation.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformationFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// already included via header:
// #include <vector>

//------------------------------------------------------------------------------
TwoBowedSurfacesDeformation::TwoBowedSurfacesDeformation(const std::vector<double> &pars)
  : theParameters(pars)
{
  if (pars.size() != minParameterSize()) {
    edm::LogError("BadSetup") << "@SUB=TwoBowedSurfacesDeformation"
                              << "Input vector of wrong size " << pars.size()
                              << " instead of " << minParameterSize() << ", add zeros to fill up!";
  }
  while (theParameters.size() < minParameterSize()) theParameters.push_back(0.);
}

//------------------------------------------------------------------------------
TwoBowedSurfacesDeformation* TwoBowedSurfacesDeformation::clone() const
{
  return new TwoBowedSurfacesDeformation(theParameters);
}

//------------------------------------------------------------------------------
int TwoBowedSurfacesDeformation::type() const
{
  return SurfaceDeformationFactory::kTwoBowedSurfaces;
}

//------------------------------------------------------------------------------
SurfaceDeformation::Local2DVector 
TwoBowedSurfacesDeformation::positionCorrection(const Local2DPoint &localPos,
						const LocalTrackAngles &localAngles,
						double length, double width) const
{
  const double ySplit = this->parameters().back();

// treatment of different widthes at high/low y could be done by theRelWidthLowY or so
//   if (widthLowY > 0. && widthHighY != widthLowY) {
//     std::cout << "SurfaceDeformation::positionCorrection2Bowed: Cannot yet deal "
// 	      << " with different widthes, take " << widthHighY << " not " << widthLowY
// 	      << std::endl;
//   }
//   const double width = widthHighY;
  
  // Some signs depend on whether we are in surface part below or above ySplit:
  const double sign = (localPos.y() < ySplit ? +1. : -1.); 
  const double yMiddle = ySplit * 0.5 - sign * length * .25;
  // 'calibrate' y length and transform y to be w.r.t. surface middle
  const double myY = localPos.y() - yMiddle;
  const double myLength = length * 0.5 + sign * ySplit;
  
  double uRel = 2. * localPos.x() / width;  // relative u (-1 .. +1)
  double vRel = 2. * myY / myLength;        // relative v (-1 .. +1)
  // 'range check':
  const double cutOff = 1.5;
  if (uRel < -cutOff) { uRel = -cutOff; } else if (uRel > cutOff) { uRel = cutOff; }
  if (vRel < -cutOff) { vRel = -cutOff; } else if (vRel > cutOff) { vRel = cutOff; }
  
  const std::vector<double> &pars = this->parameters();
  // 1st, get dw effect depending 
  // - on the surface sagittas (Legendre polynomials),
  //   see BowedSurfaceAlignmentDerivatives::operator()(..)
  // - relative dw
  // - surface specific dalpha (note that this shifts surface specific dw)
  // - surface specific dbeta
  const double dw 
    = (uRel * uRel - 1./3.) * (pars[0] + sign * pars[9])  // sagittaX
    +  uRel * vRel          * (pars[1] + sign * pars[10]) // sagittaXY
    + (vRel * vRel - 1./3.) * (pars[2] + sign * pars[11]) // sagittaY
    + sign * pars[5]                 // different dw
    + myY          * sign * pars[6]  // different dalpha
    - localPos.x() * sign * pars[7]; // different dbeta
  // 2nd, translate the dw effect to shifts in x and y
  // Positive dxdz/dydz and positive dw mean negative shift in x/y: 
  Local2DVector::ScalarType x = -dw * localAngles.dxdz();
  Local2DVector::ScalarType y = -dw * localAngles.dydz();
  // 3rd, treat in-plane differences depending on surface from xy-shifts... 
  x += (sign * pars[3]); // different du
  y += (sign * pars[4]); // different dv
  //     ...and gamma-rotation
  x -= myY          * (sign * pars[8]); // different dgamma for u
  y += localPos.x() * (sign * pars[8]); // different dgamma for v

  return Local2DVector(x, y);
}

//------------------------------------------------------------------------------
bool TwoBowedSurfacesDeformation::add(const SurfaceDeformation &other)
{
  if (this->type() == other.type()) {
    const std::vector<double> otherParameters(other.parameters());
    if (otherParameters.size() == theParameters.size()) {
      if (theParameters.back() == otherParameters.back()) {
	for (unsigned int i = 0; i < theParameters.size() - 1; ++i) {// -1 for ySplit
	  // mean bows, delta shifts, delta angles and delta bows can simply be added up
	  theParameters[i] += otherParameters[i];
	}
	return true;
      } else { // ySplit values are different!
	LogDebug("Alignment") << "@SUB=TwoBowedSurfacesDeformation::add"
			      << "Different ySplit: this " << theParameters.back() 
			      << ", to add " << otherParameters.back();
      }
    } // same size
  } // same type

  return false;
}
  
//------------------------------------------------------------------------------
std::vector<double> TwoBowedSurfacesDeformation::parameters() const
{
  return theParameters;
}
