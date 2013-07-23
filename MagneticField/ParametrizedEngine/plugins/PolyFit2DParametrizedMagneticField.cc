/** \file
 *
 *  $Date: 2011/04/16 10:20:40 $
 *  $Revision: 1.1 $
 *  \author N. Amapane
 */

#include "PolyFit2DParametrizedMagneticField.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "BFit.h"


using namespace std;
using namespace magfieldparam;

PolyFit2DParametrizedMagneticField::PolyFit2DParametrizedMagneticField(double bVal) : 
  theParam(new BFit())
{
  theParam->SetField(bVal);
}


PolyFit2DParametrizedMagneticField::PolyFit2DParametrizedMagneticField(const edm::ParameterSet& parameters) : theParam(new BFit()) {
  theParam->SetField(parameters.getParameter<double>("BValue"));
}


PolyFit2DParametrizedMagneticField::~PolyFit2DParametrizedMagneticField() {
  delete theParam;
}


GlobalVector
PolyFit2DParametrizedMagneticField::inTesla(const GlobalPoint& gp) const {

  if (isDefined(gp)) {
    return inTeslaUnchecked(gp);
  } else {
    edm::LogWarning("MagneticField|FieldOutsideValidity") << " Point " << gp << " is outside the validity region of PolyFit2DParametrizedMagneticField";
    return GlobalVector();
  }
}

GlobalVector
PolyFit2DParametrizedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const {
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
PolyFit2DParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  double z = fabs(gp.z());
  double r = gp.perp();
  //"rectangle" |z|<3.5, r<1.9 _except_ the "corners" |z|+2.5*r>6.7, everything in meters
  if (z>350. || r>190 || z+2.5*r>670.) return false;
  return true;
}
