#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionBarrelEstimator.h"

// Field
#include "MagneticField/Engine/interface/MagneticField.h"
//
#include "CLHEP/Matrix/Matrix.h"
// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
//
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
//
#include "RecoTracker/TkNavigation/interface/StartingLayerFinder.h"
#include "RecoTracker/TkNavigation/interface/LayerCollector.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h" 

//
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Geometry/Point3D.h"




InOutConversionSeedFinder::~InOutConversionSeedFinder() {
  std::cout << " InOutConversionSeedFinder DTOR " << std::endl;

}


