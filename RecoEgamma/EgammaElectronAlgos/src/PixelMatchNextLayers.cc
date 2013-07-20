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
// $Id: PixelMatchNextLayers.cc,v 1.18 2013/01/02 18:59:12 dlange Exp $
//
//

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h" 
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchNextLayers.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <iostream> 
#include <algorithm>

 
PixelMatchNextLayers::PixelMatchNextLayers(const LayerMeasurements * theLayerMeasurements,const DetLayer* ilayer, 
					   FreeTrajectoryState & aFTS,
					   const PropagatorWithMaterial *aProp,  
					   const BarrelMeasurementEstimator *aBarrelMeas,
					   const ForwardMeasurementEstimator *aForwardMeas,
					   const TrackerTopology *tTopo,
					   bool searchInTIDTEC)
 {

  typedef std::vector<TrajectoryMeasurement>::const_iterator aMeas;
  std::vector<const DetLayer*> allayers;
  std::vector<const DetLayer*> nl = ilayer->nextLayers( aFTS, alongMomentum);
  for (std::vector<const DetLayer*>::const_iterator il = nl.begin(); il != nl.end(); il++) {
    allayers.push_back(*il);
    std::vector<const DetLayer*> n2l = (*il)->nextLayers( aFTS, alongMomentum);
    for (std::vector<const DetLayer*>::const_iterator i2l = n2l.begin(); i2l != n2l.end(); i2l++) {
      allayers.push_back(*i2l);
    }
  }

  const TrajectoryStateOnSurface tsos(aFTS,ilayer->surface());
  
  if (tsos.isValid()) 
    {    
      for (std::vector<const DetLayer*>::const_iterator il = allayers.begin(); il != allayers.end(); il++) 
	{
	  if ( (*il)->subDetector()==GeomDetEnumerators::PixelBarrel || (*il)->subDetector()==GeomDetEnumerators::PixelEndcap ) {
	    
	    std::vector<TrajectoryMeasurement> pixelMeasurements;
	    if ((*il)->subDetector()==GeomDetEnumerators::PixelBarrel) {
	      pixelMeasurements = theLayerMeasurements->measurements( **il, tsos , *aProp, *aBarrelMeas); 
	    } else {
	      pixelMeasurements = theLayerMeasurements->measurements( **il, tsos, *aProp, *aForwardMeas);
	    }
	    for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++){
	      if (m == pixelMeasurements.begin()){
		CLHEP::Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
				      m->forwardPredictedState().globalPosition().y(),
				      m->forwardPredictedState().globalPosition().z());
		predictionHere.push_back( prediction);
	      }
	      if (m->recHit()->isValid()) {
		measurementsHere.push_back( *m);
		hitsHere.push_back( m->recHit());

		//std::cout<<"\n SH B-D "<<std::endl;

	      } else {
		badMeasurementsHere.push_back( *m);
	      }
	    }
	  }
	  if (searchInTIDTEC) {	  
	  //additional search in the TID layers
	  if ( ((*il)->subDetector())==GeomDetEnumerators::TID && (ilayer->location()) == GeomDetEnumerators::endcap)
	    {
	      std::vector<TrajectoryMeasurement> pixelMeasurements;
	      pixelMeasurements = theLayerMeasurements->measurements( (**il), tsos , *aProp, *aForwardMeas); 
	      
	      for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++)
		{
		  // limit search in first ring
		  if (tTopo->tidRing(m->recHit()->geographicalId()) > 1) continue;
		  if (m == pixelMeasurements.begin())
		    {
		      CLHEP::Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
					    m->forwardPredictedState().globalPosition().y(),
					    m->forwardPredictedState().globalPosition().z());
		      predictionHere.push_back( prediction);
		    }
		  if (m->recHit()->isValid()) 
		    {
		      measurementsHere.push_back( *m);
		      hitsHere.push_back(m->recHit());
		    }
		  // else{ std::cout<<" 2H not valid "<<std::endl;}
		}
	    } //end of TID search
	  
	  //additional search in the TEC layers
	  if ( ((*il)->subDetector())==GeomDetEnumerators::TEC && (ilayer->location()) == GeomDetEnumerators::endcap)
	    {
	      std::vector<TrajectoryMeasurement> pixelMeasurements;
	      pixelMeasurements = theLayerMeasurements->measurements( (**il), tsos , *aProp, *aForwardMeas); 
	      	      
	      for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++)
		{
		  // limit search in first ring and first third wheels
		  if (tTopo->tecRing(m->recHit()->geographicalId()) > 1) continue;
		  if (tTopo->tecWheel(m->recHit()->geographicalId()) > 3) continue;
		  if (m == pixelMeasurements.begin())
		    {
		      CLHEP::Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
					    m->forwardPredictedState().globalPosition().y(),
					    m->forwardPredictedState().globalPosition().z());
		      predictionHere.push_back( prediction);
		    }
		  if (m->recHit()->isValid()) 
		    {
		      measurementsHere.push_back( *m);
		      hitsHere.push_back(m->recHit());

		      //std::cout<<"\n SH TEC "<<std::endl;

		    }
		  // else{ std::cout<<" 2H not valid "<<std::endl;}
		}
	    } //end of TEC search
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

TransientTrackingRecHit::RecHitContainer PixelMatchNextLayers::hitsInNextLayers() const {

  return hitsHere;
}

std::vector<CLHEP::Hep3Vector> PixelMatchNextLayers::predictionInNextLayers() const {

  return predictionHere;
}










