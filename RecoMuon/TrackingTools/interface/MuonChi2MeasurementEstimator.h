#ifndef RecoMuon_TrackingTools_MuonChi2MeasurementEstimator_H
#define RecoMuon_TrackingTools_MuonChi2MeasurementEstimator_H

/** \class MuonChi2MeasurementEstimator
 *  Class to handle different chi2 cut parameters for each muon sub-system.
 *  MuonChi2MeasurementEstimator inherits from the Chi2MeasurementEstimatorBase class and uses
 *  3 different estimators.
 *
 *  $Date: 2008/07/24 16:40:08 $
 *  $Revision: 1.3 $
 *  \author Giorgia Mila - INFN Torino
 */


#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"


class MuonChi2MeasurementEstimator : public Chi2MeasurementEstimatorBase {
 public:
  
  /// Constructor detector indipendent
  MuonChi2MeasurementEstimator(double maxChi2, double nSigma = 3.);
  
  /// Constructor detector dependent
  MuonChi2MeasurementEstimator(double dtMaxChi2, double cscMaxChi2, double rpcMaxChi2, double nSigma);
  
  /// Chi2 estimator
  virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
					  const TransientTrackingRecHit&) const;


 private:
  
  Chi2MeasurementEstimator theDTChi2Estimator;
  Chi2MeasurementEstimator theCSCChi2Estimator;
  Chi2MeasurementEstimator theRPCChi2Estimator;


};

#endif
