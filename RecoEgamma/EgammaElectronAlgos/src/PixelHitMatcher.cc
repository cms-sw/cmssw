// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      PixelHitMatcher
// 
/**\class PixelHitMatcher EgammaElectronAlgos/PixelHitMatcher

 Description: central class for finding compatible hits

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: PixelHitMatcher.cc,v 1.10 2008/02/14 22:24:40 charlot Exp $
//
//

#include "DataFormats/Math/interface/Point3D.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchNextLayers.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;
using namespace std;

PixelHitMatcher::~PixelHitMatcher()
{ 
  delete prop1stLayer;
  delete prop2ndLayer;
  delete theLayerMeasurements;
}

void PixelHitMatcher::setES(const MagneticField* magField, const MeasurementTracker *theMeasurementTracker){
  theGeometricSearchTracker=theMeasurementTracker->geometricSearchTracker();
  startLayers.setup(theGeometricSearchTracker);
  theLayerMeasurements = new LayerMeasurements(theMeasurementTracker);
  theMagField = magField;
  delete prop2ndLayer;
  float mass=.1057; // using muon propagation
  prop1stLayer = new PropagatorWithMaterial(oppositeToMomentum,mass,theMagField);
  prop2ndLayer = new PropagatorWithMaterial(alongMomentum,mass,theMagField);
}

vector<pair<RecHitWithDist, PixelHitMatcher::ConstRecHitPointer> > PixelHitMatcher::compatibleHits(const GlobalPoint& xmeas,
												   const GlobalPoint& vprim,
										  float energy,
										  float fcharge) {

  vertex=vprim.z();
  
  int charge = int(fcharge);
  // return all compatible RecHit pairs (vector< TSiPixelRecHit>)
  vector<pair<RecHitWithDist, ConstRecHitPointer> > result;
  LogDebug("") << "[PixelHitMatcher::compatibleHits] entering .. ";


  vector<TrajectoryMeasurement> validMeasurements;
  vector<TrajectoryMeasurement> invalidMeasurements;

  typedef vector<TrajectoryMeasurement>::const_iterator aMeas;

  pred1Meas.clear();
  pred2Meas.clear();


  typedef vector<BarrelDetLayer*>::const_iterator BarrelLayerIterator;
  BarrelLayerIterator firstLayer = startLayers.firstBLayer();

  FreeTrajectoryState fts =myFTS(theMagField,xmeas, vprim, 
				 energy, charge);

  PerpendicularBoundPlaneBuilder bpb;
  TrajectoryStateOnSurface tsos(fts, *bpb(fts.position(), fts.momentum()));
  
  if (tsos.isValid()) {
    vector<TrajectoryMeasurement> pixelMeasurements = 
      theLayerMeasurements->measurements(**firstLayer,tsos, 
					 *prop1stLayer, meas1stBLayer);
 
    LogDebug("") <<"[PixelHitMatcher::compatibleHits] nbr of hits compatible with extrapolation to first layer: " << pixelMeasurements.size();
    for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++){
     if (m->recHit()->isValid()) {
	Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
			      m->forwardPredictedState().globalPosition().y(),
			      m->forwardPredictedState().globalPosition().z());
         LogDebug("") << "[PixelHitMatcher::compatibleHits] compatible hit position " << m->recHit()->globalPosition();
         LogDebug("") << "[PixelHitMatcher::compatibleHits] predicted position " << m->forwardPredictedState().globalPosition();
	pred1Meas.push_back( prediction);
      
	validMeasurements.push_back(*m);
	LogDebug("") <<"[PixelHitMatcher::compatibleHits] Found a rechit in layer ";
	const BarrelDetLayer *bdetl = dynamic_cast<const BarrelDetLayer *>(*firstLayer);
	if (bdetl) {
	  LogDebug("") <<" with radius "<<bdetl->specificSurface().radius();
	}
	else  LogDebug("") <<"Could not downcast!!";
      } 
    }

    
    // check if there are compatible 1st hits in the second layer
    firstLayer++;

    vector<TrajectoryMeasurement> pixel2Measurements = 
      theLayerMeasurements->measurements(**firstLayer,tsos,
					 *prop1stLayer, meas1stBLayer);
 
    for (aMeas m=pixel2Measurements.begin(); m!=pixel2Measurements.end(); m++){
      if (m->recHit()->isValid()) {
        Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
			      m->forwardPredictedState().globalPosition().y(),
			      m->forwardPredictedState().globalPosition().z());
	pred1Meas.push_back( prediction);
        LogDebug("")  << "[PixelHitMatcher::compatibleHits] compatible hit position " << m->recHit()->globalPosition() << endl;
        LogDebug("") << "[PixelHitMatcher::compatibleHits] predicted position " << m->forwardPredictedState().globalPosition() << endl;
      
	validMeasurements.push_back(*m);
	LogDebug("") <<"[PixelHitMatcher::compatibleHits] Found a rechit in layer ";
	const BarrelDetLayer *bdetl = dynamic_cast<const BarrelDetLayer *>(*firstLayer);
	if (bdetl) {
	  LogDebug("") <<" with radius "<<bdetl->specificSurface().radius();
	}
	else  LogDebug("") <<"Could not downcast!!";
      }
    }
  }

 
  // check if there are compatible 1st hits the forward disks
  typedef vector<ForwardDetLayer*>::const_iterator ForwardLayerIterator;
  ForwardLayerIterator flayer;
  
  TrajectoryStateOnSurface tsosfwd(fts, *bpb(fts.position(), fts.momentum()));  
  if (tsosfwd.isValid()) {

    for (int i=0; i<2; i++) {
      i == 0 ? flayer = startLayers.pos1stFLayer() : flayer = startLayers.neg1stFLayer();
      vector<TrajectoryMeasurement> pixelMeasurements = 
	theLayerMeasurements->measurements(**flayer, tsosfwd,
					   *prop1stLayer, meas1stFLayer);


      for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++){
	if (m->recHit()->isValid()) {
	  Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
				m->forwardPredictedState().globalPosition().y(),
				m->forwardPredictedState().globalPosition().z());
	  pred1Meas.push_back( prediction);
	
	  validMeasurements.push_back(*m);      
	}
      }
    }
  }

  // now we have the vector of all valid measurements of the first point
  for (unsigned i=0; i<validMeasurements.size(); i++){
    const DetLayer* newLayer = theGeometricSearchTracker->detLayer(validMeasurements[i].recHit()->det()->geographicalId());

    // compute the z vertex from the cluster point and the found pixel hit
    double pxHit1z = validMeasurements[i].recHit()->det()->surface().toGlobal(
									      validMeasurements[i].recHit()->localPosition()).z();
//    double pxHit1r = validMeasurements[i].recHit()->det()->surface().toGlobal(
//									      validMeasurements[i].recHit()->localPosition()).perp();
    double pxHit1x = validMeasurements[i].recHit()->det()->surface().toGlobal(
									      validMeasurements[i].recHit()->localPosition()).x();
    double pxHit1y = validMeasurements[i].recHit()->det()->surface().toGlobal(
									      validMeasurements[i].recHit()->localPosition()).y();
       
//    double zVertexPred = pxHit1z - pxHit1r*(xmeas.z()-pxHit1z)/
//      (xmeas.perp()-pxHit1r);
      
//    double pxHit1phi = validMeasurements[i].recHit()->det()->surface().toGlobal(
//									     validMeasurements[i].recHit()->localPosition()).phi();
  
    double r1diff = (pxHit1x-vprim.x())*(pxHit1x-vprim.x()) + (pxHit1y-vprim.y())*(pxHit1y-vprim.y());
    r1diff=sqrt(r1diff);
    double r2diff = (xmeas.x()-pxHit1x)*(xmeas.x()-pxHit1x) + (xmeas.y()-pxHit1y)*(xmeas.y()-pxHit1y);
    r2diff=sqrt(r2diff);
    double zVertexPred = pxHit1z - r1diff*(xmeas.z()-pxHit1z)/r2diff;

//    double zVertexRec = vertex;
//    std::cout << "vertex new calculation " <<  zVertexPred << " and reco vertex " << zVertexRec << std::endl;

    GlobalPoint vertexPred(vprim.x(),vprim.y(),zVertexPred);
    vertex = zVertexPred;
    
    // evaluated z vertex is needed for the forward estimator
    meas2ndFLayer.setVertex(vertex);
    
    GlobalPoint hitPos( validMeasurements[i].recHit()->det()->surface().toGlobal(
										 validMeasurements[i].recHit()->localPosition())); 

    FreeTrajectoryState secondFTS=myFTS(theMagField,hitPos,vertexPred,energy,charge);

    PixelMatchNextLayers secondHit(theLayerMeasurements, newLayer, secondFTS,
				   prop2ndLayer, &meas2ndBLayer,&meas2ndFLayer);
    vector<Hep3Vector> predictions = secondHit.predictionInNextLayers();

    for (unsigned it = 0; it < predictions.size(); it++) pred2Meas.push_back(predictions[it]); 

    // we may get more than one valid second measurements here even for single electrons: 
    // two hits from the same layer/disk (detector overlap) or from the loop over the
    // next layers in EPMatchLoopNextLayers. Take only the 1st hit.    
    if(!secondHit.measurementsInNextLayers().empty()){
      float dphi = pred1Meas[i].phi()-validMeasurements[i].recHit()->globalPosition().phi();
      if (dphi > pi) dphi -= twopi;
      if (dphi < -pi) dphi += twopi; 

      ConstRecHitPointer pxrh = validMeasurements[i].recHit();
      
      RecHitWithDist rh(pxrh,dphi);
      
      pxrh = secondHit.measurementsInNextLayers()[0].recHit();
      pair<RecHitWithDist, ConstRecHitPointer> compatiblePair = pair<RecHitWithDist, ConstRecHitPointer>(rh,pxrh);
      result.push_back(compatiblePair);
    }

    //We may have one layer left, try that, if no valid hits
    if(secondHit.measurementsInNextLayers().empty()){
      vector<TrajectoryMeasurement> missedMeasurements = secondHit.badMeasurementsInNextLayers();
      for (unsigned j=0; j<missedMeasurements.size();j++){
        if (!missedMeasurements[j].recHit()->det()) continue;
        const DetLayer* newLayer = theGeometricSearchTracker->detLayer(missedMeasurements[j].recHit()->det()->geographicalId());
	PixelMatchNextLayers secondSecondHit(theLayerMeasurements, newLayer, secondFTS,
					     prop2ndLayer, &meas2ndBLayer,&meas2ndFLayer);

        vector<Hep3Vector> predictions = secondSecondHit.predictionInNextLayers();

        for (unsigned it = 0; it < predictions.size(); it++) pred2Meas.push_back(predictions[it]); 
	
        if(!secondSecondHit.measurementsInNextLayers().empty()){
	  float dphi = pred1Meas[i].phi()-validMeasurements[i].recHit()->globalPosition().phi();
	  if (dphi > pi) dphi -= twopi;
	  if (dphi < -pi) dphi += twopi; 
	  ConstRecHitPointer pxrh = validMeasurements[i].recHit();
	  RecHitWithDist rh(pxrh,dphi);
	  pxrh = secondSecondHit.measurementsInNextLayers()[0].recHit();
	  pair<RecHitWithDist, ConstRecHitPointer> compatiblePair = pair<RecHitWithDist, ConstRecHitPointer>(rh,pxrh);
          result.push_back(compatiblePair);
        }
      }
    }
  }

  return result;

}


vector<Hep3Vector> PixelHitMatcher::predicted1Hits() {

  return pred1Meas;
}

vector<Hep3Vector> PixelHitMatcher::predicted2Hits() {

  return pred2Meas;
}

float PixelHitMatcher::getVertex(){

  return vertex;
}





