#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
//


// Field
#include "MagneticField/Engine/interface/MagneticField.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h" 

//



ConversionSeedFinder::ConversionSeedFinder(const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker) :
  theMF_(field), theMeasurementTracker_(theInputMeasurementTracker ), 
  theOutwardStraightPropagator_(theMF_, dir_ = alongMomentum ),
  thePropagatorWithMaterial_(dir_ = alongMomentum, 0.000511, theMF_ ), theUpdator_()

 
{

  std::cout << " ConversionSeedFinder CTOR " << std::endl;
      
};
