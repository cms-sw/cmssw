#ifndef MuonChi2MeasurementEstimator_H
#define MuonChi2MeasurementEstimator_H


/** \class MuonChi2MeasurementEstimator
 *  A Chi2 Measurement Estimator.
 *  Starts from a transientTrack recHit.
 *  The Chi2 can have different values associated to different muon subdetectors.
 *
 *  $Date: 2008/07/22 09:57:27 $
 *  $Revision: 1.1 $
 *  \author Giorgia Mila - INFN Torino
 */


#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"


class MuonChi2MeasurementEstimator : public Chi2MeasurementEstimatorBase {
 public:
  
  /// Constructor detector indipendent
  MuonChi2MeasurementEstimator(double maxChi2, double nSigma);
  
  /// Constructor detector dependent
  MuonChi2MeasurementEstimator(double dtMaxChi2, double cscMaxChi2, double rpcMaxChi2, double nSigma);
  
  /// Chi2 asociator
  std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				  const TransientTrackingRecHit&) const;


 private:
  
  Chi2MeasurementEstimator theDtChi2Estimator;
  Chi2MeasurementEstimator theCscChi2Estimator;
  Chi2MeasurementEstimator theRpcChi2Estimator;


};

#endif
