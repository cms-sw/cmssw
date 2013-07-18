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
// $Id: PixelMatchNextLayers.h,v 1.9 2009/05/27 14:08:25 fabiocos Exp $
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
class TrackerTopology;

class PixelMatchNextLayers {

public:
  PixelMatchNextLayers(const LayerMeasurements * theLayerMeasurements, const DetLayer* ilayer, FreeTrajectoryState & aFTS,
	                        const PropagatorWithMaterial *aProp, 
		       const BarrelMeasurementEstimator *aBarrelMeas,
		       const ForwardMeasurementEstimator *aForwardMeas,
		       const TrackerTopology *tTopo,
		       bool searchInTIDTEC);
  std::vector<TrajectoryMeasurement> measurementsInNextLayers() const;
  std::vector<TrajectoryMeasurement> badMeasurementsInNextLayers() const;
  //RC vector<TSiPixelRecHit> hitsInNextLayers() const;  
  //In this way we are losing the information about the kind of the ReferenceCounted TTRH? 
  TransientTrackingRecHit::RecHitContainer hitsInNextLayers() const;  
  std::vector<CLHEP::Hep3Vector> predictionInNextLayers() const;

  
private:
                                                        
  std::vector<TrajectoryMeasurement> measurementsHere;
  std::vector<TrajectoryMeasurement> badMeasurementsHere;  
  TransientTrackingRecHit::RecHitContainer hitsHere;
  std::vector<CLHEP::Hep3Vector> predictionHere; 
};

#endif




