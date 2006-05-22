#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H

/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/05/19 15:24:35 $
 *  $Revision: 1.4 $
 *  \author R. Bellan - INFN Torino
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"

// FIXME
// change FreeTrajectoryState in TSOS

namespace edm {class ParameterSet; class EventSetup;}

class StandAloneMuonRefitter {
public:
  /// Constructor
  StandAloneMuonRefitter(const edm::ParameterSet& par);

  /// Destructor
  virtual ~StandAloneMuonRefitter(){};

  // Operations
  
  /// Perform the inner-outward fitting
  void refit(FreeTrajectoryState& initialState);

  /// the last free trajectory state
  FreeTrajectoryState lastUpdatedFTS() const {return *theLastUpdatedTSOS.freeTrajectoryState();}

  /// the last but one free trajectory state
  FreeTrajectoryState lastButOneUpdatedFTS() const {return *theLastButOneUpdatedTSOS.freeTrajectoryState();}
  
  /// the Trajectory state on the last surface of the fitting
  TrajectoryStateOnSurface lastUpdatedTSOS() const {return theLastUpdatedTSOS;}

  /// the Trajectory state on the last surface of the fitting
  TrajectoryStateOnSurface lastButOneUpdatedTSOS() const {return theLastButOneUpdatedTSOS;}

  void reset();
  void setES(const edm::EventSetup& setup);

  int getTotalChamberUsed() const {return totalChambers;}
  int getDTChamberUsed() const {return dtChambers;}
  int getCSCChamberUsed() const {return cscChambers;}
  int getRPCChamberUsed() const {return rpcChambers;}


protected:

private:

  /// Set the last TSOS
  void setLastUpdatedTSOS(TrajectoryStateOnSurface tsos) { theLastUpdatedTSOS = tsos;}
  
  /// Set the last but one TSOS
  void setLastButOneUpdatedTSOS(TrajectoryStateOnSurface tsos) { theLastButOneUpdatedTSOS = tsos;}
  

  /// the trajectory state on the last available surface
  TrajectoryStateOnSurface theLastUpdatedTSOS;
  /// the trajectory state on the last but one available surface
  TrajectoryStateOnSurface theLastButOneUpdatedTSOS;

  // The Measurement extractor
  MuonDetLayerMeasurements theMeasureExtractor;

  int totalChambers;
  int dtChambers;
  int cscChambers;
  int rpcChambers;
};
#endif

