#include "MagneticField/Engine/interface/MagneticField.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"

MagneticFieldMap*
MagneticFieldMap::myself=0; 

MagneticFieldMap* 
MagneticFieldMap::instance(const MagneticField* pMF) {
  if (!myself) myself = new MagneticFieldMap(pMF);
  return myself;
}

MagneticFieldMap* 
MagneticFieldMap::instance() {
  return myself;
}


const GlobalVector
MagneticFieldMap::inTesla( const GlobalPoint& gp) const {

  if (!instance()) {
    return GlobalVector( 0., 0., 4.);
  } else {
    return pMF_->inTesla(gp);
  }

}

const GlobalVector 
MagneticFieldMap::inKGauss( const GlobalPoint& gp) const {
  
  return inTesla(gp) * 10.;

}

const GlobalVector
MagneticFieldMap::inInverseGeV( const GlobalPoint& gp) const {
  
  return inKGauss(gp) * 2.99792458e-4;

} 

double 
MagneticFieldMap::inTeslaZ(const GlobalPoint& gp) const {

    return instance() ? pMF_->inTesla(gp).z() : 4.0;

}

double 
MagneticFieldMap::inKGaussZ(const GlobalPoint& gp) const {

    return inTeslaZ(gp)/10.;

}

double 
MagneticFieldMap::inInverseGeVZ(const GlobalPoint& gp) const {

   return inKGaussZ(gp) * 2.99792458e-4;

}

