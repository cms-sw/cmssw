#ifndef MuonChi2MeasurementEstimator_H
#define MuonChi2MeasurementEstimator_H


/** \class MuonChi2MeasurementEstimator
 *  A Chi2 Measurement Estimator.
 *  Starts from a transientTrack recHit.
 *  The Chi2 can have different values associated to different muon subdetectors.
 *
 *  $Date: 2007/05/09 14:05:13 $
 *  $Revision: 1.3 $
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
  Chi2MeasurementEstimator estimate(const TransientTrackingRecHit&) const;


 private:
  
  Chi2MeasurementEstimator theDtChi2Estimator;
  Chi2MeasurementEstimator theCscChi2Estimator;
  Chi2MeasurementEstimator theRpcChi2Estimator;


};

#endif
