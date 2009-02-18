/** \class MuonUpdatorAtVertex
 *  This class do the extrapolation of a TrajectoryStateOnSurface to the PCA and can apply, with a different
 *  method, the vertex constraint. The vertex constraint is applyed using the Kalman Filter tools used for 
 *  the vertex reconstruction.
 *
 *  $Date: 2009/02/18 18:22:26 $
 *  $Revision: 1.36 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

/// Constructor
MuonUpdatorAtVertex::MuonUpdatorAtVertex(const edm::ParameterSet& pset,
					 const MuonServiceProxy *service):theService(service){
   
  thePropagatorName = pset.getParameter<string>("Propagator");
    
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
MuonUpdatorAtVertex::~MuonUpdatorAtVertex(){}

// Operations

/// Propagate the state to the PCA in 2D, i.e. to the beam line
pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::propagate(const TrajectoryStateOnSurface &tsos, const reco::BeamSpot & beamSpot) const{

  const string metname = "Muon|RecoMuon|MuonUpdatorAtVertex";
 
  if(TrackerBounds::isInside(tsos.globalPosition())){
    LogTrace(metname) << "Trajectory inside the Tracker";

    TrajectoryStateClosestToBeamLineBuilder tscblBuilder;
    TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(*(tsos.freeState()),
							  beamSpot);

    if(tscbl.isValid())
      return pair<bool,FreeTrajectoryState>(true,tscbl.trackStateAtPCA());
    else
      edm::LogWarning(metname) << "Propagation to the PCA using TSCPBuilderNoMaterial failed!"
			       << " This can cause a severe bug.";
  }
  else{
    LogTrace(metname) << "Trajectory inside the muon system";

    FreeTrajectoryState
      result =  theService->propagator(thePropagatorName)->propagate(*tsos.freeState(),beamSpot);
    
    LogTrace(metname) << "MuonUpdatorAtVertex::propagate, path: "
		      << result << " parameters: " << result.parameters();
    
    if(result.hasError()) 
      return pair<bool,FreeTrajectoryState>(true,result);
    else
      edm::LogInfo(metname) << "Propagation to the PCA failed!";
  }
  return pair<bool,FreeTrajectoryState>(false,FreeTrajectoryState());
}


pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::update(const reco::TransientTrack & track, const reco::BeamSpot & beamSpot) const{

  const std::string metname = "Muon|RecoMuon|MuonUpdatorAtVertex";
    
  
  pair<bool,FreeTrajectoryState> result(false,FreeTrajectoryState());
  
  SingleTrackVertexConstraint::TrackFloatPair constrainedTransientTrack;

  try{
    GlobalPoint spotPos(beamSpot.x0(),beamSpot.y0(),beamSpot.z0());
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
MuonUpdatorAtVertex::update(const FreeTrajectoryState& ftsAtVtx, const reco::BeamSpot & beamSpot) const{
  
  return update(theTransientTrackFactory.build(ftsAtVtx),beamSpot);
}


pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::propagateWithUpdate(const TrajectoryStateOnSurface &tsos, const reco::BeamSpot & beamSpot) const{
  
  pair<bool,FreeTrajectoryState>
    propagationResult = propagate(tsos,beamSpot);

  if(propagationResult.first){
    return update(propagationResult.second, beamSpot);
  }
  else{
    edm::LogInfo("Muon|RecoMuon|MuonUpdatorAtVertex") << "Constraint at vertex failed";
    return pair<bool,FreeTrajectoryState>(false,FreeTrajectoryState());
  }
}


 std::pair<bool,FreeTrajectoryState>
 MuonUpdatorAtVertex::propagateToNominalLine(const TrajectoryStateOnSurface &tsos) const{
   
   const string metname = "Muon|RecoMuon|MuonUpdatorAtVertex";
   
   if(TrackerBounds::isInside(tsos.globalPosition())){
     LogTrace(metname) << "Trajectory inside the Tracker";
     
     TSCPBuilderNoMaterial tscpBuilder;
     TrajectoryStateClosestToPoint tscp = tscpBuilder(*(tsos.freeState()),
						     GlobalPoint(0.,0.,0.));
    
    if(tscp.isValid())
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
      result = theService->propagator(thePropagatorName)->propagateWithPath(*tsos.freeState(),p1,p2);
    
    LogTrace(metname) << "MuonUpdatorAtVertex::propagate, path: "
		      << result.second << " parameters: " << result.first.parameters();
    
    if(result.first.hasError()) 
      return pair<bool,FreeTrajectoryState>(true,result.first);
    else
      edm::LogInfo(metname) << "Propagation to the PCA failed! Path: "<<result.second;
  }
  return pair<bool,FreeTrajectoryState>(false,FreeTrajectoryState());  
}

std::pair<bool,FreeTrajectoryState>
MuonUpdatorAtVertex::propagate(const TrajectoryStateOnSurface &tsos) const{
  return propagateToNominalLine(tsos);
}
