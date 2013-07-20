#ifndef RecoMuon_TrackingTools_MuonUpdatorAtVertex_H
#define RecoMuon_TrackingTools_MuonUpdatorAtVertex_H

/** \class MuonUpdatorAtVertex
 *  This class do the extrapolation of a TrajectoryStateOnSurface to the PCA and can apply, with a different
 *  method, the vertex constraint. The vertex constraint is applyed using the Kalman Filter tools used for 
 *  the vertex reconstruction.
 *
 *  For the time being the propagator is the SteppingHelixPropagator because the method propagate(TSOS,GlobalPoint)
 *  it is in its specific interface. Once the interface of the Propagator base class will be updated, 
 *  then propagator will become generic. 
 *
 *  For what concern the beam spot, it is possible set via cff the relevant parameters:
 *
 *  BeamSpotPosition[0] <=> x
 *  BeamSpotPosition[1] <=> y
 *  BeamSpotPosition[2] <=> z
 *
 *  BeamSpotPositionErrors[0] = sigma(x) 
 *  BeamSpotPositionErrors[1] = sigma(y) 
 *  BeamSpotPositionErrors[2] = sigma(z)
 *
 *
 *  $Date: 2009/09/16 13:12:07 $
 *  $Revision: 1.22 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

class TrajectoryStateOnSurface;
class FreeTrajectoryState;
class MuonServiceProxy;

#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <string>

namespace edm {class ParameterSet; class Event;}

class MuonUpdatorAtVertex {
public:
  /// Constructor
  MuonUpdatorAtVertex(const edm::ParameterSet& pset, const MuonServiceProxy *service);

  /// Destructor
  virtual ~MuonUpdatorAtVertex();

  // Operations
  
  /// Propagate the state to the 2D-PCA
  std::pair<bool,FreeTrajectoryState>
    propagate(const TrajectoryStateOnSurface &tsos, const reco::BeamSpot & beamSpot) const;

  /// Applies the vertex constraint
  std::pair<bool,FreeTrajectoryState> 
    update(const reco::TransientTrack &track, const reco::BeamSpot & beamSpot) const;
  
  /// Applies the vertex constraint
  std::pair<bool,FreeTrajectoryState>
    update(const FreeTrajectoryState& ftsAtVtx, const reco::BeamSpot & beamSpot) const;

  /// Propagate to the 2D-PCA and apply the vertex constraint
  std::pair<bool,FreeTrajectoryState>
    propagateWithUpdate(const TrajectoryStateOnSurface &tsos,
			const reco::BeamSpot & beamSpot) const;

  /// Propagate the state to the 2D-PCA (nominal CMS axis)
  std::pair<bool,FreeTrajectoryState>
    propagateToNominalLine(const TrajectoryStateOnSurface &tsos) const;

  /// Propagate the state to the 2D-PCA (nominal CMS axis) - DEPRECATED -
  std::pair<bool,FreeTrajectoryState>
    propagate(const TrajectoryStateOnSurface &tsos) const __attribute__((deprecated));

  

protected:

private:

  const MuonServiceProxy *theService;
  std::string thePropagatorName;
 
  TransientTrackFromFTSFactory theTransientTrackFactory;
  SingleTrackVertexConstraint theConstrictor;
  double theChi2Cut;

  GlobalError thePositionErrors;
};
#endif

