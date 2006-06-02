#ifndef PIXELMATCHNEXTLAYERS_H
#define PIXELMATCHNEXTLAYERS_H
// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      PixelMatchNextLayers
// 
/**\class PixelMatchNextLayers EgammaElectronAlgos/PixelMatchNextLayers

 Description: class to find the compatible hits in the next layer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id$
//
//
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "CLHEP/Vector/ThreeVector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h"
#include <vector>

class DetLayer;
class FreeTrajectoryState;
class PropagatorWithMaterial;
class LayerMeasurements;

class PixelMatchNextLayers {

public:
  PixelMatchNextLayers(const LayerMeasurements * theLayerMeasurements, const DetLayer* ilayer, FreeTrajectoryState & aFTS,
	                        const PropagatorWithMaterial *aProp, 
                      const BarrelMeasurementEstimator *aBarrelMeas,
		      const ForwardMeasurementEstimator *aForwardMeas);
  vector<TrajectoryMeasurement> measurementsInNextLayers() const;
  vector<TrajectoryMeasurement> badMeasurementsInNextLayers() const;
  vector<TSiPixelRecHit> hitsInNextLayers() const;
  vector<Hep3Vector> predictionInNextLayers() const;

  
private:
                                                        
  vector<TrajectoryMeasurement> measurementsHere;
  vector<TrajectoryMeasurement> badMeasurementsHere;
  vector<TSiPixelRecHit> hitsHere;
  vector<Hep3Vector> predictionHere; 
};

#endif




