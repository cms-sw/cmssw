#ifndef FastSimulation_ParticlePropagator_MagneticFieldMap_H
#define FastSimulation_ParticlePropagator_MagneticFieldMap_H

// Framework Headers
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

// Famos headers
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"

#include <map>
#include <string>

class MagneticField;
class TrackerInteractionGeometry;
class TH1;

class MagneticFieldMap {

public:

  const GlobalVector inTesla( const GlobalPoint& ) const;
  const GlobalVector inKGauss( const GlobalPoint& ) const;
  const GlobalVector inInverseGeV( const GlobalPoint& ) const;
  const GlobalVector inTesla(const TrackerLayer& aLayer, double coord, int success) const;
  double inTeslaZ(const GlobalPoint&) const;
  double inKGaussZ(const GlobalPoint&) const;
  double inInverseGeVZ(const GlobalPoint&) const;
  double inTeslaZ(const TrackerLayer& aLayer, double coord, int success) const;

  const MagneticField& magneticField() const {return *pMF_;}

  static MagneticFieldMap* instance(const MagneticField* pMF,
				    TrackerInteractionGeometry* myGeo);

  static MagneticFieldMap* instance() ;

private:

  MagneticFieldMap(const MagneticField* pMF,
		   TrackerInteractionGeometry* myGeo);

  void initialize();

  static MagneticFieldMap* myself;
  const MagneticField* pMF_;
  TrackerInteractionGeometry* geometry_;
  std::map<int,TH1*> fieldBarrelHistos;
  std::map<int,TH1*> fieldEndcapHistos;

};

#endif // FastSimulation_ParticlePropagator_MagneticFieldMap_H
