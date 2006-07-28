#ifndef FastSimulation_ParticlePropagator_MagneticFieldMap_H
#define FastSimulation_ParticlePropagator_MagneticFieldMap_H

// Framework Headers
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
  const MagneticField& magneticField() const {return *pMF_;}

  static MagneticFieldMap* instance(const MagneticField* pMF) ;
  static MagneticFieldMap* instance() ;

private:

  MagneticFieldMap(const MagneticField* pMF) : pMF_(pMF) {;}
  static MagneticFieldMap* myself;
  const MagneticField* pMF_;

};

#endif // FastSimulation_ParticlePropagator_MagneticFieldMap_H
