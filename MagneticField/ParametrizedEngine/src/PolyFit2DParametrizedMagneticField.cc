/** \file
 *
 *  $Date: 2008/04/23 14:50:04 $
 *  $Revision: 1.2 $
 *  \author N. Amapane
 */

#include <MagneticField/ParametrizedEngine/src/PolyFit2DParametrizedMagneticField.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <MagneticField/ParametrizedEngine/src/BFit.h>


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

  double Br, Bz, Bphi;
  if (isDefined(gp)) {
    theParam->GetField(gp.perp()/100., gp.phi(), gp.z()/100.,
		       Br, Bz, Bphi);
      return GlobalVector(GlobalVector::Cylindrical(Br,Bphi,Bz));
  } else {
    edm::LogWarning("MagneticField|FieldOutsideValidity") << " Point " << gp << " is outside the validity region of PolyFit2DParametrizedMagneticField";
    return GlobalVector();
  }
}


bool
PolyFit2DParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  double z = fabs(gp.z());
  double r = gp.perp();
  //"rectangle" |z|<3.5, r<1.9 _except_ the "corners" |z|+2.5*r>6.7, everything in meters
  if (z>350. || r>190 || z+2.5*r>670.) return false;
  return true;
}
