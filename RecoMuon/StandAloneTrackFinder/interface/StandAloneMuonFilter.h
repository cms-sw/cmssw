#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonFilter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonFilter_H

/** \class StandAloneMuonFilter
 *  The inward-outward fitter (starts from seed state).
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *          D. Trocino - INFN Torino <daniele.trocino@to.infn.it>
 *
 *           Modified by C. Calabria & A. Sharma
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "RecoMuon/TrackingTools/interface/MuonBestMeasurementFinder.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class Propagator;
class DetLayer;
class MuonTrajectoryUpdator;
class Trajectory;
class MuonDetLayerMeasurements;
class MeasurementEstimator;
class MuonServiceProxy;

namespace edm {class ParameterSet; class EventSetup; class Event;}

class StandAloneMuonFilter {

 public:
    /// Constructor
  StandAloneMuonFilter(const edm::ParameterSet& par, const MuonServiceProxy* service,edm::ConsumesCollector& iC);

  /// Destructor
  virtual ~StandAloneMuonFilter();

  // Operations
  
  /// Perform the inner-outward fitting
  void refit(const TrajectoryStateOnSurface& initialState, const DetLayer*, Trajectory& trajectory);

  /// the last free trajectory state
  FreeTrajectoryState lastUpdatedFTS() const {return *theLastUpdatedTSOS.freeTrajectoryState();}

  /// the last but one free trajectory state
  FreeTrajectoryState lastButOneUpdatedFTS() const {return *theLastButOneUpdatedTSOS.freeTrajectoryState();}
  
  /// the Trajectory state on the last compatible surface
  TrajectoryStateOnSurface lastCompatibleTSOS() const {return theLastCompatibleTSOS;}

  /// the Trajectory state on the last surface of the fitting
  TrajectoryStateOnSurface lastUpdatedTSOS() const {return theLastUpdatedTSOS;}

  /// the Trajectory state on the last surface of the fitting
  TrajectoryStateOnSurface lastButOneUpdatedTSOS() const {return theLastButOneUpdatedTSOS;}

  void reset();

  /// Pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);

  int getTotalChamberUsed() const {return totalChambers;}
  int getDTChamberUsed() const {return dtChambers;}
  int getCSCChamberUsed() const {return cscChambers;}
  int getRPCChamberUsed() const {return rpcChambers;}
  int getGEMChamberUsed() const {return gemChambers;}

  int getTotalCompatibleChambers() const {return totalCompatibleChambers;}
  int getDTCompatibleChambers() const {return dtCompatibleChambers;}
  int getCSCCompatibleChambers() const {return cscCompatibleChambers;}
  int getRPCCompatibleChambers() const {return rpcCompatibleChambers;}
  int getGEMCompatibleChambers() const {return gemCompatibleChambers;}

  inline bool goodState() const {return totalChambers >= 2 && 
				   ((dtChambers + cscChambers + gemChambers) >0 ||
				    onlyRPC());}
  
  inline bool isCompatibilitySatisfied() const {return totalCompatibleChambers >= 2 && 
						  ((dtCompatibleChambers + cscCompatibleChambers + gemCompatibleChambers) >0 ||
						   onlyRPC());}
  
  /// return the layer used for the refit
  std::vector<const DetLayer*> layers() const {return theDetLayers;}

  /// return the last det layer
  const DetLayer* lastDetLayer() const {return theDetLayers.back();}

  /// Return the propagation direction
  PropagationDirection propagationDirection() const;

  /// Return the fit direction
  NavigationDirection fitDirection() const {return theFitDirection;}
  
  /// True if there are only the RPC measurements
  bool onlyRPC() const {return theRPCLoneliness;}

  /// access at the estimator
  MeasurementEstimator *estimator() const {return theEstimator;}
  
  /// access at the propagator
  const Propagator *propagator() const;
  
  /// access at the muon updator
  MuonTrajectoryUpdator *updator() const {return theMuonUpdator;}

  void createDefaultTrajectory(const Trajectory &, Trajectory &);


protected:

private:

  /// Set the last compatible TSOS
  void setLastCompatibleTSOS(TrajectoryStateOnSurface tsos) { theLastCompatibleTSOS = tsos;}
  
  /// Set the last TSOS
  void setLastUpdatedTSOS(TrajectoryStateOnSurface tsos) { theLastUpdatedTSOS = tsos;}
  
  /// Set the last but one TSOS
  void setLastButOneUpdatedTSOS(TrajectoryStateOnSurface tsos) { theLastButOneUpdatedTSOS = tsos;}
  
  /// Increment the DT,CSC,RPC counters
  void incrementChamberCounters(const DetLayer *layer);
  void incrementCompatibleChamberCounters(const DetLayer *layer);
 
  /// Set the rigth Navigation
  std::vector<const DetLayer*> compatibleLayers(const DetLayer *initialLayer,
						const FreeTrajectoryState& fts,
						PropagationDirection propDir);

  bool update(const DetLayer * layer, 
              const TrajectoryMeasurement * meas,
              Trajectory & trajectory);

  std::vector<TrajectoryMeasurement>
  findBestMeasurements(const DetLayer * layer, const TrajectoryStateOnSurface & tsos);
  
  /// the trajectory state on the last compatible surface
  TrajectoryStateOnSurface theLastCompatibleTSOS;
  /// the trajectory state on the last available surface
  TrajectoryStateOnSurface theLastUpdatedTSOS;
  /// the trajectory state on the last but one available surface
  TrajectoryStateOnSurface theLastButOneUpdatedTSOS;

  /// The Measurement extractor
  MuonDetLayerMeasurements *theMeasurementExtractor;
  
    /// The Estimator
  MeasurementEstimator *theEstimator;

  /// the muon updator (it doesn't inhert from an updator, but it has one!)
  MuonTrajectoryUpdator *theMuonUpdator;
  /// its name
  std::string theMuonUpdatorName;
  
  /// The best measurement finder: search for the best measurement among the TMs available
  MuonBestMeasurementFinder *theBestMeasurementFinder;
  /// Access to the best measurement finder
  MuonBestMeasurementFinder *bestMeasurementFinder() const {return theBestMeasurementFinder;}

  /// The max allowed chi2 to accept a rechit in the fit
  double theMaxChi2;
  /// The errors of the trajectory state are multiplied by nSigma 
  /// to define acceptance of BoundPlane and maximalLocalDisplacement
  double theNSigma;

  /// the propagation direction
  NavigationDirection theFitDirection;

  /// the det layer used in the reconstruction
  std::vector<const DetLayer*> theDetLayers;

  /// the propagator name
  std::string thePropagatorName;

  /// Navigation type
  /// "Direct","Standard"
  std::string theNavigationType;

  /// True if there are only the RPC measurements
  bool theRPCLoneliness;

  int totalChambers;
  int dtChambers;
  int cscChambers;
  int rpcChambers;
  int gemChambers;

  int totalCompatibleChambers;
  int dtCompatibleChambers;
  int cscCompatibleChambers;
  int rpcCompatibleChambers;
  int gemCompatibleChambers;
 
  const MuonServiceProxy *theService;
  bool theOverlappingChambersFlag;
};
#endif

