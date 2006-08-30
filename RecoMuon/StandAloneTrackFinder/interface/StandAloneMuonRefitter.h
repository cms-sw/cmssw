#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H

/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/07/11 15:01:07 $
 *  $Revision: 1.17 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

class Propagator;
class DetLayer;
class MuonTrajectoryUpdator;
class Trajectory;
class MuonDetLayerMeasurements;
class MeasurementEstimator;
class MuonDetLayerGeometry;
class MuonServiceProxy;

namespace edm {class ParameterSet; class EventSetup; class Event;}

class StandAloneMuonRefitter {
public:
  /// Constructor
  StandAloneMuonRefitter(const edm::ParameterSet& par, const MuonServiceProxy* service);

  /// Destructor
  virtual ~StandAloneMuonRefitter();

  // Operations
  
  /// Perform the inner-outward fitting
  void refit(const TrajectoryStateOnSurface& initialState, const DetLayer*, Trajectory& trajectory);

  /// the last free trajectory state
  FreeTrajectoryState lastUpdatedFTS() const {return *theLastUpdatedTSOS.freeTrajectoryState();}

  /// the last but one free trajectory state
  FreeTrajectoryState lastButOneUpdatedFTS() const {return *theLastButOneUpdatedTSOS.freeTrajectoryState();}
  
  /// the Trajectory state on the last surface of the fitting
  TrajectoryStateOnSurface lastUpdatedTSOS() const {return theLastUpdatedTSOS;}

  /// the Trajectory state on the last surface of the fitting
  TrajectoryStateOnSurface lastButOneUpdatedTSOS() const {return theLastButOneUpdatedTSOS;}

  void reset();

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);

  /// Pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);

  int getTotalChamberUsed() const {return totalChambers;}
  int getDTChamberUsed() const {return dtChambers;}
  int getCSCChamberUsed() const {return cscChambers;}
  int getRPCChamberUsed() const {return rpcChambers;}

  /// return the layer used for the refit
  std::vector<const DetLayer*> layers() const {return theDetLayers;}

  /// return the last det layer
  const DetLayer* lastDetLayer() const {return theDetLayers.back();}

  /// Return the propagation direction
  PropagationDirection propagationDirection() const {return thePropagationDirection;}

protected:

private:

  /// Set the last TSOS
  void setLastUpdatedTSOS(TrajectoryStateOnSurface tsos) { theLastUpdatedTSOS = tsos;}
  
  /// Set the last but one TSOS
  void setLastButOneUpdatedTSOS(TrajectoryStateOnSurface tsos) { theLastButOneUpdatedTSOS = tsos;}
  
  /// Increment the DT,CSC,RPC counters
  void incrementChamberCounters(const DetLayer *layer);

  /// Extract the Event Setup info at each event. It is called by setES
  virtual void init(const edm::EventSetup& setup);
  
  /// Set the rigth Navigation
  std::vector<const DetLayer*> compatibleLayers(const DetLayer *initialLayer,
						FreeTrajectoryState& fts,
						PropagationDirection &propDir);
  
  /// the trajectory state on the last available surface
  TrajectoryStateOnSurface theLastUpdatedTSOS;
  /// the trajectory state on the last but one available surface
  TrajectoryStateOnSurface theLastButOneUpdatedTSOS;

  /// The Measurement extractor
  MuonDetLayerMeasurements *theMeasurementExtractor;
  
  /// The propagator
  Propagator *thePropagator;
  
  /// access at the propagator
  Propagator *propagator() const {return thePropagator;}

  /// The Estimator
  MeasurementEstimator *theEstimator;

  /// access at the estimator
  MeasurementEstimator *estimator() const {return theEstimator;}

  /// the muon updator (it doesn't inhert from an updator, but it has one!)
  MuonTrajectoryUpdator *theMuonUpdator;
  /// its name
  std::string theMuonUpdatorName;
  
  /// access at the muon updator
  MuonTrajectoryUpdator *updator() const {return theMuonUpdator;}

  /// The max allowed chi2 to accept a rechit in the fit
  double theMaxChi2;
  /// The errors of the trajectory state are multiplied by nSigma 
  /// to define acceptance of BoundPlane and maximalLocalDisplacement
  double theNSigma;

  /// the propagation direction
  PropagationDirection thePropagationDirection;

  /// the det layer used in the reconstruction
  std::vector<const DetLayer*> theDetLayers;

  /// the propagator name
  std::string thePropagatorName;

  /// Navigation type
  /// "Direct","Standard"
  std::string theNavigationType;

  int totalChambers;
  int dtChambers;
  int cscChambers;
  int rpcChambers;

  MuonDetLayerGeometry *theDetLayerGeometry;
};
#endif

