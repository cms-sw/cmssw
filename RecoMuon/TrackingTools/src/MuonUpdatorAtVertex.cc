/** \class MuonUpdatorAtVertex
 *  This class do the extrapolation of a TrajectoryStateOnSurface to the PCA and can apply, with a different
 *  method, the vertex constraint. The vertex constraint is applyed using the Kalman Filter tools used for 
 *  the vertex reconstruction.
 *
 *  $Date: 2007/03/01 20:47:06 $
 *  $Revision: 1.21 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

/// Constructor
MuonUpdatorAtVertex::MuonUpdatorAtVertex(const string &propagatorName,
					 const MuonServiceProxy *service):theService(service){
  
  // FIXME
  theChi2Cut = 1000000.;

  // FIXME

  // The SteppingHelixPropagator must be used explicitly since the method propagate(TSOS,GlobalPoint)
  // is only in its specific interface. Once the interface of the Propagator base class  will be
  // updated, then thePropagator will become generic. The string and the MuonServiceProxy are used
  // in order to make more simpler and faster the future transition.
  
  thePropagator = 0;
  
  // FIXME
  // remove the flag as the Propagator base class will gains the propagate(TSOS,Position) method
  theFirstTime = true;
}

/// Destructor
MuonUpdatorAtVertex::~MuonUpdatorAtVertex(){
  delete thePropagator;
}

// Operations


/////////
// FIXME!!! remove this method as the Propagator will gains the propagate(TSOS,Position) method
// remove the flag as well
void MuonUpdatorAtVertex::setPropagator(){
  const string metname = "Muon|RecoMuon|MuonUpdatorAtVertex";
  
  if(theFirstTime ||
     theService->isTrackingComponentsRecordChanged()){
    if(thePropagator) delete thePropagator;
    Propagator *propagator = &*theService->propagator("SteppingHelixPropagatorOpposite")->clone();
    thePropagator = dynamic_cast<SteppingHelixPropagator*>(propagator);  
    theFirstTime = false;

    LogTrace(metname) << " MuonUpdatorAtVertex::setPropagator: propagator changed!";
  }
  
}
///////////

/// Propagate the state to the vertex
// FIXME it is const. It will be when setPropagator() will be removed
pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::propagate(const TrajectoryStateOnSurface &tsos, 
			       const GlobalPoint &vtxPosition){

  const string metname = "Muon|RecoMuon|MuonUpdatorAtVertex";

  setPropagator();
  //  return thePropagator->propagate(*tsos.freeState(),vtxPosition);
  pair<FreeTrajectoryState,double> 
    result = thePropagator->propagateWithPath(*tsos.freeState(),vtxPosition);

  LogTrace(metname) << "MuonUpdatorAtVertex::propagate, path: "
		    << result.second << " parameters: " << result.first.parameters();

  if( result.first.hasError()) 
    return pair<bool,FreeTrajectoryState>(true,result.first);
  else{
    edm::LogWarning(metname) << "Propagation to the PCA failed!";
    
    // FIXME: returns FreeTrajectoryState() instead of result.first?
    return pair<bool,FreeTrajectoryState>(false,result.first);
  }
}


/// Propagate the state to the PCA in 2D, i.e. to the beam line
// FIXME it is const. It will be when setPropagator() will be removed
pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::propagate(const TrajectoryStateOnSurface &tsos){

  const string metname = "Muon|RecoMuon|MuonUpdatorAtVertex";

  setPropagator();

  if(TrackerBounds::isInside(tsos.globalPosition())){
    
    TSCPBuilderNoMaterial tscpBuilder;
    TrajectoryStateClosestToPoint tscp = tscpBuilder(*(tsos.freeState()),
						     GlobalPoint(0.,0.,0.)); //FIXME Correct?

    // FIXME: check if the tscp is valid or not!!
    if(tscp.hasError())
      return pair<bool,FreeTrajectoryState>(true,tscp.theState());
    else
      edm::LogWarning(metname) << "Propagation to the PCA using TSCPBuilderNoMaterial failed!"
			       << " This can cause a severe bug.";
  }
  else{
    // Define a line using two 3D-points
    GlobalPoint p1(0.,0.,-1500);
    GlobalPoint p2(0.,0.,1500);
    
    pair<FreeTrajectoryState,double> 
      result = thePropagator->propagateWithPath(*tsos.freeState(),p1,p2);
    
    LogTrace(metname) << "MuonUpdatorAtVertex::propagate, path: "
		      << result.second << " parameters: " << result.first.parameters();
    
    if(result.first.hasError()) 
      return pair<bool,FreeTrajectoryState>(true,result.first);
    else{
      edm::LogWarning(metname) << "Propagation to the PCA failed!";
      
      // FIXME: returns FreeTrajectoryState() instead of result.first?
      return pair<bool,FreeTrajectoryState>(false,result.first);
    }
  }
}


// FIXME it is const. It will be when setPropagator() will be removed
pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::update(const reco::TransientTrack & track){

  const std::string metname = "Muon|RecoMuon|MuonUpdatorAtVertex";
    
  // FIXME
  setPropagator();

  pair<bool,FreeTrajectoryState> result(false,FreeTrajectoryState());
  
  GlobalPoint glbPos(0.,0.,0.);
  
  // assume beam spot position with nominal errors
  // sigma(x) = sigma(y) = 15 microns
  // sigma(z) = 5.3 cm

  AlgebraicSymMatrix mat(3,0);
  mat[0][0] = (15.e-04)*(15.e-04);
  mat[1][1] = (15.e-04)*(15.e-04);
  mat[2][2] = (5.3)*(5.3);
  GlobalError glbErrPos(mat);

  SingleTrackVertexConstraint::TrackFloatPair constrainedTransientTrack;

  try{
    constrainedTransientTrack = theConstrictor.constrain(track,glbPos, glbErrPos);
  }
  catch ( cms::Exception& e ) {
    edm::LogWarning(metname) << "cms::Exception caught in MuonUpdatorAtVertex::update\n"
			     << e.explainSelf();
    return result;
  }

  if(constrainedTransientTrack.second <= theChi2Cut) {
    result.first = true;
    result.second = *constrainedTransientTrack.first.impactPointState().freeState();
  }
  else
    edm::LogWarning(metname) << "Constraint at vertex failed"; 
    
  return result;
}

pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::update(const FreeTrajectoryState& ftsAtVtx){
  
  return update(theTransientTrackFactory.build(ftsAtVtx));
}



pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::propagateWithUpdate(const TrajectoryStateOnSurface &tsos, 
					 const GlobalPoint &vtxPosition){
  
  pair<bool,FreeTrajectoryState>
    propagationResult = propagate(tsos,vtxPosition);

  if(propagationResult.first){
    // FIXME!!!
    // This is very very temporary! Waiting for the changes in the KalmanVertexFitter interface
    return update(propagationResult.second);
  }
  else{
    edm::LogWarning("Muon|RecoMuon|MuonUpdatorAtVertex") << "Constraint at vertex failed";
    return pair<bool,FreeTrajectoryState>(false,FreeTrajectoryState());
  }
}


pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::propagateWithUpdate(const TrajectoryStateOnSurface &tsos){
  
  pair<bool,FreeTrajectoryState>
    propagationResult = propagate(tsos);

  if(propagationResult.first){
    // FIXME!!!
    // This is very very temporary! Waiting for the changes in the KalmanVertexFitter interface
    return update(propagationResult.second);
  }
  else{
    edm::LogWarning("Muon|RecoMuon|MuonUpdatorAtVertex") << "Constraint at vertex failed";
    return pair<bool,FreeTrajectoryState>(false,FreeTrajectoryState());
  }
}
