#ifndef FastSimulation_ParticlePropagator_MagneticFieldMap_H
#define FastSimulation_ParticlePropagator_MagneticFieldMap_H

// Framework Headers
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

class MagneticField;

class MagneticFieldMap {

public:

  const GlobalVector inTesla( const GlobalPoint& ) const;
  const GlobalVector inKGauss( const GlobalPoint& ) const;
  const GlobalVector inInverseGeV( const GlobalPoint& ) const;
  double inTeslaZ(const GlobalPoint&) const;
  double inKGaussZ(const GlobalPoint&) const;
  double inInverseGeVZ(const GlobalPoint&) const;

  static MagneticFieldMap* instance(edm::ESHandle<MagneticField> pMF) ;
  static MagneticFieldMap* instance() ;

private:

  MagneticFieldMap(edm::ESHandle<MagneticField> pMF) : pMF_(pMF) {;}
  static MagneticFieldMap* myself;
  edm::ESHandle<MagneticField> pMF_;

};

#endif // FastSimulation_ParticlePropagator_MagneticFieldMap_H
