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
// $Id: PixelMatchNextLayers.cc,v 1.4 2007/02/05 17:53:52 uberthon Exp $
//
//

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h" 
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchNextLayers.h"
#include <iostream> 
#include <algorithm>

 
PixelMatchNextLayers::PixelMatchNextLayers(const LayerMeasurements * theLayerMeasurements,const DetLayer* ilayer, 
                      FreeTrajectoryState & aFTS,
					     const PropagatorWithMaterial *aProp,  
                      const BarrelMeasurementEstimator *aBarrelMeas,
		      const ForwardMeasurementEstimator *aForwardMeas) {

  typedef std::vector<TrajectoryMeasurement>::const_iterator aMeas;
  std::vector<const DetLayer*> nl = ilayer->nextLayers( aFTS, alongMomentum);
  const TrajectoryStateOnSurface tsos(aFTS,ilayer->surface());
  for (std::vector<const DetLayer*>::const_iterator il = nl.begin(); il != nl.end(); il++) {

    //    if ( (*il)->module()==pixel) {
    if ( (*il)->subDetector()==GeomDetEnumerators::PixelBarrel || (*il)->subDetector()==GeomDetEnumerators::PixelEndcap ) {

      std::vector<TrajectoryMeasurement> pixelMeasurements;
      if ((*il)->location() == GeomDetEnumerators::barrel) {
  	pixelMeasurements = theLayerMeasurements->measurements( **il, tsos , *aProp, *aBarrelMeas); 
      } else {
	pixelMeasurements = theLayerMeasurements->measurements( **il, tsos, *aProp, *aForwardMeas);
      }
      for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++){
        if (m == pixelMeasurements.begin()){
          Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
                                m->forwardPredictedState().globalPosition().y(),
                                m->forwardPredictedState().globalPosition().z());
          predictionHere.push_back( prediction);
        }
        if (m->recHit()->isValid()) {
	  //	  cout << "    Found compatible hit" << endl;
          measurementsHere.push_back( *m);
	  //RC const TSiPixelRecHit *hit= dynamic_cast<const TSiPixelRecHit*>(m->recHit());
	  const TSiPixelRecHit *hit= dynamic_cast<const TSiPixelRecHit*>(m->recHit().get());
	  //RC if (hit) hitsHere.push_back( *hit);
	  if (hit) hitsHere.push_back( m->recHit());
	} else {
          badMeasurementsHere.push_back( *m);
	}
      }
    }
  } 
}

  
std::vector<TrajectoryMeasurement> PixelMatchNextLayers::measurementsInNextLayers() const {

  return measurementsHere;
}

std::vector<TrajectoryMeasurement> PixelMatchNextLayers::badMeasurementsInNextLayers() const {

  return badMeasurementsHere;
}

//RC vector<TSiPixelRecHit> PixelMatchNextLayers::hitsInNextLayers() const {
TransientTrackingRecHit::RecHitContainer PixelMatchNextLayers::hitsInNextLayers() const {

  return hitsHere;
}

std::vector<Hep3Vector> PixelMatchNextLayers::predictionInNextLayers() const {

  return predictionHere;
}










