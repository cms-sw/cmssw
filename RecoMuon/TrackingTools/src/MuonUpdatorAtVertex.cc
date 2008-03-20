/** \class MuonUpdatorAtVertex
 *  This class do the extrapolation of a TrajectoryStateOnSurface to the PCA and can apply, with a different
 *  method, the vertex constraint. The vertex constraint is applyed using the Kalman Filter tools used for 
 *  the vertex reconstruction.
 *
 *  $Date: 2008/02/20 08:48:24 $
 *  $Revision: 1.30 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

/// Constructor
MuonUpdatorAtVertex::MuonUpdatorAtVertex(const edm::ParameterSet& pset,
					 const MuonServiceProxy *service):theService(service){
  
  // FIXME
  // The SteppingHelixPropagator must be used explicitly since the method propagate(TSOS,GlobalPoint)
  // is only in its specific interface. Once the interface of the Propagator base class  will be
  // updated, then thePropagator will become generic. The string and the MuonServiceProxy are used
  // in order to make more simpler and faster the future transition.
  
  thePropagatorName = pset.getParameter<string>("Propagator");
  thePropagator = 0;
  
  // FIXME
  // remove the flag as the Propagator base class will gains the propagate(TSOS,Position) method
  theFirstTime = true;
  
  // Position of the beam spot
  vector<double> position = pset.getParameter< vector<double> >("BeamSpotPosition");
  if(position.size() != 3) 
    edm::LogError("Muon|RecoMuon|MuonUpdatorAtVertex")
      <<"MuonUpdatorAtVertex::BeamSpotPosition wrong number of parameters!!";
  
  // assume:
  // position[0] <=> x
  // position[1] <=> y
  // position[2] <=> z
  GlobalPoint glbPos(position[0],position[1],position[2]);
  thePosition = glbPos;
  
  // Errors on the Beam spot position
  vector<double> errors = pset.getParameter< vector<double> >("BeamSpotPositionErrors");
  if(errors.size() != 3) 
    edm::LogError("Muon|RecoMuon|MuonUpdatorAtVertex")
      <<"MuonUpdatorAtVertex::BeamSpotPositionErrors wrong number of parameters!!";
  
  // assume:
  // errors[0] = sigma(x) 
  // errors[1] = sigma(y) 
  // errors[2] = sigma(z)

  AlgebraicSymMatrix33 mat;
  mat(0,0) = errors[0]*errors[0];
  mat(1,1) = errors[1]*errors[1];
  mat(2,2) = errors[2]*errors[2];
  GlobalError glbErrPos(mat);

  thePositionErrors = glbErrPos;

  // cut on chi^2
  theChi2Cut = pset.getParameter<double>("MaxChi2");
}

/// Destructor
MuonUpdatorAtVertex::~MuonUpdatorAtVertex(){
  if (thePropagator) delete thePropagator;
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
    Propagator *propagator = &*theService->propagator(thePropagatorName)->clone();
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
    edm::LogInfo(metname) << "Propagation to the PCA failed!";
    
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
    LogTrace(metname) << "Trajectory inside the Tracker";

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
    LogTrace(metname) << "Trajectory inside the muon system";

    // Define a line using two 3D-points
    GlobalPoint p1(0.,0.,-1500);
    GlobalPoint p2(0.,0.,1500);
    
    pair<FreeTrajectoryState,double> 
      result = thePropagator->propagateWithPath(*tsos.freeState(),p1,p2);
    
    LogTrace(metname) << "MuonUpdatorAtVertex::propagate, path: "
		      << result.second << " parameters: " << result.first.parameters();
    
    if(result.first.hasError()) 
      return pair<bool,FreeTrajectoryState>(true,result.first);
    else
      edm::LogInfo(metname) << "Propagation to the PCA failed! Path: "<<result.second;
  }
  return pair<bool,FreeTrajectoryState>(false,FreeTrajectoryState());
}


// FIXME it is const. It will be when setPropagator() will be removed
pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::update(const reco::TransientTrack & track, edm::Event &event){

  const std::string metname = "Muon|RecoMuon|MuonUpdatorAtVertex";
    
  // FIXME
  setPropagator();

  pair<bool,FreeTrajectoryState> result(false,FreeTrajectoryState());
  
  SingleTrackVertexConstraint::TrackFloatPair constrainedTransientTrack;

  try{
    edm::Handle<reco::BeamSpot> beamSpot;
    event.getByType(beamSpot);

    GlobalPoint spotPos(beamSpot->x0(),beamSpot->y0(),beamSpot->z0());
    constrainedTransientTrack = theConstrictor.constrain(track, spotPos, thePositionErrors);
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
    edm::LogInfo(metname) << "Constraint at vertex failed"; 
    
  return result;
}

pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::update(const FreeTrajectoryState& ftsAtVtx, edm::Event &event){
  
  return update(theTransientTrackFactory.build(ftsAtVtx),event);
}



pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::propagateWithUpdate(const TrajectoryStateOnSurface &tsos, 
					 const GlobalPoint &vtxPosition,
					 edm::Event &event){
  
  pair<bool,FreeTrajectoryState>
    propagationResult = propagate(tsos,vtxPosition);

  if(propagationResult.first){
    // FIXME!!!
    // This is very very temporary! Waiting for the changes in the KalmanVertexFitter interface
    return update(propagationResult.second,event);
  }
  else{
    edm::LogInfo("Muon|RecoMuon|MuonUpdatorAtVertex") << "Constraint at vertex failed";
    return pair<bool,FreeTrajectoryState>(false,FreeTrajectoryState());
  }
}


pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::propagateWithUpdate(const TrajectoryStateOnSurface &tsos, edm::Event &event){
  
  pair<bool,FreeTrajectoryState>
    propagationResult = propagate(tsos);

  if(propagationResult.first){
    // FIXME!!!
    // This is very very temporary! Waiting for the changes in the KalmanVertexFitter interface
    return update(propagationResult.second,event);
  }
  else{
    edm::LogInfo("Muon|RecoMuon|MuonUpdatorAtVertex") << "Constraint at vertex failed";
    return pair<bool,FreeTrajectoryState>(false,FreeTrajectoryState());
  }
}
