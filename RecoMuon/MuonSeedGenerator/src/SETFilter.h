#ifndef RecoMuon_MuonSeedGenerator_SETFilter_H
#define RecoMuon_MuonSeedGenerator_SETFilter_H

/** \class SETFilter
    I. Bloch, E. James, S. Stoynev
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

//#include "CLHEP/Matrix/DiagMatrix.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Vector/ThreeVector.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"


// used in the SET algorithm
struct SeedCandidate{
  MuonTransientTrackingRecHit::MuonRecHitContainer theSet;
  CLHEP::Hep3Vector momentum;
  int charge;
  double weight;
  Trajectory::DataContainer trajectoryMeasurementsInTheSet;
};

class DetLayer;
class Trajectory;
//class MuonServiceProxy;
class TrajectoryFitter;

namespace edm {class ParameterSet; class EventSetup; class Event;}

class SETFilter {

 public:
    /// Constructor
  SETFilter(const edm::ParameterSet& par, const MuonServiceProxy* service);

  /// Destructor
  virtual ~SETFilter();

  // Operations
  
  /// Perform the inner-outward fitting
  void refit(const TrajectoryStateOnSurface& initialState, const DetLayer*, Trajectory& trajectory);

  /// the last free trajectory state
  FreeTrajectoryState lastUpdatedFTS() const {return *theLastUpdatedTSOS.freeTrajectoryState();}

  /// the Trajectory state on the last surface of the fitting
  TrajectoryStateOnSurface lastUpdatedTSOS() const {return theLastUpdatedTSOS;}

  /// Perform the SET inner-outward fitting
  bool fwfit_SET( std::vector < SeedCandidate> & validSegmentsSet_in,
                           std::vector < SeedCandidate> & validSegmentsSet_out);
  
  ///  from SeedCandidate to DataContainer only
  bool buildTrajectoryMeasurements( SeedCandidate * validSegmentsSet,
				    Trajectory::DataContainer & finalCandidate);


  ///  transforms "segment trajectory" to "rechit container"  
  bool transform(Trajectory::DataContainer &measurements_segments,
                 TransientTrackingRecHit::ConstRecHitContainer & hitContainer, 
		 TrajectoryStateOnSurface & firstTSOS);

  ///  transforms "segment trajectory" to "segment container" 
  bool transformLight(Trajectory::DataContainer &measurements_segments,
		      TransientTrackingRecHit::ConstRecHitContainer & hitContainer, 
		      TrajectoryStateOnSurface & firstTSOS);



  void reset();

  /// Pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);

  int getTotalChamberUsed() const {return totalChambers;}
  int getDTChamberUsed() const {return dtChambers;}
  int getCSCChamberUsed() const {return cscChambers;}
  int getRPCChamberUsed() const {return rpcChambers;}

  inline bool goodState() const {return totalChambers >= 2 && 
				   ((dtChambers + cscChambers) >0 );}
  
  /// return the layer used for the refit
  std::vector<const DetLayer*> layers() const {return theDetLayers;}

  /// return the last det layer
  const DetLayer* lastDetLayer() const {return theDetLayers.back();}

  /// Return the propagation direction
  PropagationDirection propagationDirection() const;

protected:

private:

  /// Set the last TSOS
  void setLastUpdatedTSOS(TrajectoryStateOnSurface tsos) { theLastUpdatedTSOS = tsos;}
  
  /// Set the last but one TSOS
  void setLastButOneUpdatedTSOS(TrajectoryStateOnSurface tsos) { theLastButOneUpdatedTSOS = tsos;}
  
  /// Increment the DT,CSC,RPC counters
  void incrementChamberCounters(const DetLayer *layer);
 
  /// access at the propagator
  const Propagator *propagator() const;
  

  // chi2 functions (calculate chi2)
  double findChi2(double pX, double pY, double pZ,
                    const CLHEP::Hep3Vector& r3T,
                    SeedCandidate & muonCandidate,
                    TrajectoryStateOnSurface  &lastUpdatedTSOS,
                    Trajectory::DataContainer & trajectoryMeasurementsInTheSet,
                    bool detailedOutput);

  double findMinChi2(unsigned int iSet, const CLHEP::Hep3Vector& r3T,
                 SeedCandidate & muonCandidate,
                 std::vector < TrajectoryStateOnSurface > &lastUpdatedTSOS_Vect,
                 Trajectory::DataContainer & trajectoryMeasurementsInTheSet);

  double chi2AtSpecificStep(CLHEP::Hep3Vector &foot,
                            const CLHEP::Hep3Vector& r3T,
                            SeedCandidate & muonCandidate,
                            TrajectoryStateOnSurface  &lastUpdatedTSOS,
                            Trajectory::DataContainer & trajectoryMeasurementsInTheSet,
                            bool detailedOutput);

  // find initial points for the SIMPLEX minimization
  std::vector <CLHEP::Hep3Vector> find3MoreStartingPoints(CLHEP::Hep3Vector &key_foot,
                                                   const CLHEP::Hep3Vector& r3T,
                                                   SeedCandidate & muonCandidate);

  std::pair <double,double> findParabolaMinimum(std::vector <double> &quadratic_var,
                                                std::vector <double> &quadratic_chi2);

  // SIMPLEX minimization functions
  void pickElements(std::vector <double> &chi2Feet,
                   unsigned int & high, unsigned int & second_high, unsigned int & low);

  CLHEP::Hep3Vector reflectFoot(std::vector <CLHEP::Hep3Vector> & feet,
                         unsigned int key_foot, double scale );

  void nDimContract(std::vector <CLHEP::Hep3Vector> & feet, unsigned int low);
  //---- SET
  
  /// the propagator name
  std::string thePropagatorName;

  /// the propagation direction
  NavigationDirection theFitDirection;

  /// the trajectory state on the last available surface
  TrajectoryStateOnSurface theLastUpdatedTSOS;
  /// the trajectory state on the last but one available surface
  TrajectoryStateOnSurface theLastButOneUpdatedTSOS;

  /// the det layer used in the reconstruction
  std::vector<const DetLayer*> theDetLayers;

  int totalChambers;
  int dtChambers;
  int cscChambers;
  int rpcChambers;

  bool useSegmentsInTrajectory;

  /// used in the SET BW fit
  edm::ESHandle<TrajectoryFitter> theBWLightFitter;
  std::string theBWLightFitterName;

  const MuonServiceProxy *theService;
  //bool theOverlappingChambersFlag;
};
#endif

