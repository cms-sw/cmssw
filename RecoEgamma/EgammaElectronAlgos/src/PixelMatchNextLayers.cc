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

  typedef vector<TrajectoryMeasurement>::const_iterator aMeas;

  //UB for test ----------------------------------------------------------
//   std::endl;
//   std::cout <<" +++++++++++++ PixelMatchNextLayers called for Layer ";
//   const BarrelDetLayer *bindetl = dynamic_cast<const BarrelDetLayer *>(ilayer);
//   if (bindetl) {
//     std::cout <<" barrel with radius  "<<bindetl->specificSurface().radius()<<std::endl;
//   }
//   else   {
//     const ForwardDetLayer *findetl = dynamic_cast<const ForwardDetLayer *>(ilayer);
//     if (findetl) 
//       std::cout <<" forward with position   "<<findetl->initialPosition()<<std::endl;
//   }

//   vector<const DetLayer*> nl= ilayer->nextLayers( aFTS, oppositeToMomentum);

//   cout << "    Found " << nl.size() << " compatible layers opp"  << endl;

//   if (nl.size()) {
//     const BarrelDetLayer *bdetl = dynamic_cast<const BarrelDetLayer *>(nl[0]);
//     if (bdetl) 
//       std::cout <<"PixelMatchNextLayers: radius opposite  "<<bdetl->specificSurface().radius()<<std::endl;
//   }
  vector<const DetLayer*> nl = ilayer->nextLayers( aFTS, alongMomentum);
//  cout << "    Found " << nl.size() << " compatible layers along"  << endl;
//   const BarrelDetLayer *bdetl = dynamic_cast<const BarrelDetLayer *>(nl[0]);
//   if (bdetl) 
//     std::cout <<"PixelMatchNextLayers: radius along  "<<bdetl->specificSurface().radius()<<std::endl;
  
  //UB end test ------------------------------------
  const TrajectoryStateOnSurface tsos(aFTS,ilayer->surface());
  for (vector<const DetLayer*>::const_iterator il = nl.begin(); il != nl.end(); il++) {

    if ( (*il)->module()==pixel) {

      vector<TrajectoryMeasurement> pixelMeasurements;
      if ((*il)->part() == barrel) {
	//	cout << "      Now in barrel layer "  << endl;
// 	const BarrelDetLayer *bdetl = dynamic_cast<const BarrelDetLayer *>(*il);
// 	if (bdetl) {
// 	  std::cout <<"PixelMatchNextLayers: radius "<<bdetl->specificSurface().radius()<<std::endl;
// 	}
  	pixelMeasurements = theLayerMeasurements->measurements( **il, tsos , *aProp, *aBarrelMeas); 
      } else {
	//	std::cout << "      Now in forward disk z: " << std::endl;
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
	  const TSiPixelRecHit *hit= dynamic_cast<const TSiPixelRecHit*>(m->recHit());
	  if (hit) hitsHere.push_back( *hit);
	} else {
          badMeasurementsHere.push_back( *m);
	}
      }
    }
  } 
}

  
vector<TrajectoryMeasurement> PixelMatchNextLayers::measurementsInNextLayers() const {

  return measurementsHere;
}

vector<TrajectoryMeasurement> PixelMatchNextLayers::badMeasurementsInNextLayers() const {

  return badMeasurementsHere;
}

vector<TSiPixelRecHit> PixelMatchNextLayers::hitsInNextLayers() const {

  return hitsHere;
}

vector<Hep3Vector> PixelMatchNextLayers::predictionInNextLayers() const {

  return predictionHere;
}










