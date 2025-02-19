/** \file
 *
 *  $Date: 2011/04/16 10:20:40 $
 *  $Revision: 1.1 $
 *  \author N. Amapane
 */

#include "PolyFit3DParametrizedMagneticField.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "BFit3D.h"


using namespace std;
using namespace magfieldparam;

PolyFit3DParametrizedMagneticField::PolyFit3DParametrizedMagneticField(double bVal) : 
  theParam(new BFit3D())
{
  theParam->SetField(bVal);
}


PolyFit3DParametrizedMagneticField::PolyFit3DParametrizedMagneticField(const edm::ParameterSet& parameters) : theParam(new BFit3D()) {
  theParam->SetField(parameters.getParameter<double>("BValue"));

  // Additional options (documentation by Vassili):

  // By default, the package accepts signed value of "r". That means,
  // one can cross r=0 and orientation of the coordinate "orts"
  // e_r and e_phi will not be flipped over.
  // In other words for an r<0 the e_r points inward, in the direction r=0.
  // This is a "natural" mode. However, the default behavior may be
  // changed by the call

  //  theParam->UseSignedRad(false);
  
  // In that case with crossing of r=0 e_r and e_phi will be flipped in
  // such a way that e_r always points outward. In other words instead of 
  // (r<0, phi) the (abs(r), phi+PI) will be used in this mode.

  // The expansion coefficients for a nominal field in between measurement
  // field values (2.0T, 3.5T, 3.8T and 4.0T) by default are calculated by
  // means of a linear piecewise interpolation. Another provided
  // interpolation mode is cubic spline. This mode can be switched
  // on by the call:

  //  theParam->UseSpline(true);

  // From practical point of view the use of spline interpolation doesn't
  // change much, but it makes the coefficients' behavior a bit more
  // physical at very low/high field values.
}


PolyFit3DParametrizedMagneticField::~PolyFit3DParametrizedMagneticField() {
  delete theParam;
}


GlobalVector
PolyFit3DParametrizedMagneticField::inTesla(const GlobalPoint& gp) const {

  if (isDefined(gp)) {
    return inTeslaUnchecked(gp);
  } else {
    edm::LogWarning("MagneticField|FieldOutsideValidity") << " Point " << gp << " is outside the validity region of PolyFit3DParametrizedMagneticField";
    return GlobalVector();
  }
}

GlobalVector
PolyFit3DParametrizedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const {
  double Br, Bz, Bphi;
  theParam->GetField(gp.perp()/100., gp.z()/100., gp.phi(),
		     Br, Bz, Bphi);

  double cosphi = cos(gp.phi());
  double sinphi = sin(gp.phi());

  return GlobalVector(Br*cosphi - Bphi*sinphi,
		      Br*sinphi + Bphi*cosphi, 
		      Bz);  
}

bool
PolyFit3DParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  double z = fabs(gp.z());
  double r = gp.perp();
  //"rectangle" |z|<3.5, r<1.9 _except_ the "corners" |z|+2.5*r>6.7, everything in meters
  if (z>350. || r>190 || z+2.5*r>670.) return false;
  return true;
}
