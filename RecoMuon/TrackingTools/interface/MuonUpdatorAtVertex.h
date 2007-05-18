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
 *  $Date: 2007/02/18 16:01:47 $
 *  $Revision: 1.13 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

class TrajectoryStateOnSurface;
class FreeTrajectoryState;
class SteppingHelixPropagator;
class MuonServiceProxy;

#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <string>


class MuonUpdatorAtVertex {
public:
  /// Constructor
  MuonUpdatorAtVertex(const std::string &propagatorName, const MuonServiceProxy *service);

  /// Destructor
  virtual ~MuonUpdatorAtVertex();

  // Operations
  
  /// Propagate the state to the 3D-PCA
  std::pair<bool,FreeTrajectoryState>
    propagate(const TrajectoryStateOnSurface &tsos, 
	      const GlobalPoint &vtxPosition);
  
  /// Propagate the state to the 2D-PCA
  std::pair<bool,FreeTrajectoryState>
    propagate(const TrajectoryStateOnSurface &tsos);

  /// Applies the vertex constraint
  std::pair<bool,FreeTrajectoryState> 
    update(const reco::TransientTrack &track);
  
  /// Applies the vertex constraint
  std::pair<bool,FreeTrajectoryState>
    update(const FreeTrajectoryState& ftsAtVtx);

  /// Propagate to the 3D-PCA and apply the vertex constraint
  std::pair<bool,FreeTrajectoryState>
    propagateWithUpdate(const TrajectoryStateOnSurface &tsos, 
			const GlobalPoint &vtxPosition);
  
  /// Propagate to the 2D-PCA and apply the vertex constraint
  std::pair<bool,FreeTrajectoryState>
    propagateWithUpdate(const TrajectoryStateOnSurface &tsos);

protected:

private:

  const MuonServiceProxy *theService;

  // FIXME
  // The SteppingHelixPropagator must be used explicitly since the method propagate(TSOS,GlobalPoint)
  // is only in its specific interface. Once the interface of the Propagator base class  will be
  // updated, then thePropagator will become generic. 
  SteppingHelixPropagator *thePropagator;

  // FIXME
  // remove the flag as the Propagator base class will gains the propagate(TSOS,Position) method
  bool theFirstTime;
  
  // FIXME
  // remove this method as the Propagator will gains the propagate(TSOS,Position) method
  void setPropagator();

  TransientTrackFromFTSFactory theTransientTrackFactory;
  SingleTrackVertexConstraint theConstrictor;
  double theChi2Cut;
};
#endif

