#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonSmoother_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonSmoother_H

/** \class StandAloneMuonSmoother
 *  The outward-inward fitter (starts from StandAloneMuonBackwardFilter innermost state).
 *
 *  $Date: 2006/08/31 18:28:04 $
 *  $Revision: 1.7 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// #include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class MuonServiceProxy;
class TrajectoryStateUpdator;
class MeasurementEstimator;
class Propagator;

#include <string>

namespace edm {class ParameterSet; class Event;}

//class StandAloneMuonSmoother: public KFTrajectorySmoother {
class StandAloneMuonSmoother{

  typedef std::pair<bool,Trajectory> SmoothingResult;

 public:
  /// Constructor
  StandAloneMuonSmoother(const edm::ParameterSet& par, const MuonServiceProxy* service);

  /// Destructor
  virtual ~StandAloneMuonSmoother();
  
  // Operations 

  /// Smoothes the trajectories
  SmoothingResult smooth(const Trajectory&);
  
  /// return the KFUpdator
  TrajectoryStateUpdator *updator() const {return theUpdator;}

  /// access at the propagator
  const Propagator *propagator() const;

  /// access at the estimator
  MeasurementEstimator *estimator() const {return theEstimator;}
    
 protected:
  
 private:
  std::string thePropagatorName;

  /// The max allowed chi2 to accept a rechit in the fit
  double theMaxChi2;

  /// The errors of the trajectory state are multiplied by nSigma 
  /// to define acceptance of BoundPlane and maximalLocalDisplacement
  double theNSigma;

  /// The estimator: makes the decision wheter a measure is good or not
  /// it isn't used by the updator which does the real fit. In fact, in principle,
  /// a looser request onto the measure set can be requested 
  /// (w.r.t. the request on the accept/reject measure in the fit)
  MeasurementEstimator *theEstimator;

  double theErrorRescaling;

  TrajectoryStateUpdator *theUpdator;

  const MuonServiceProxy* theService;

};
#endif

