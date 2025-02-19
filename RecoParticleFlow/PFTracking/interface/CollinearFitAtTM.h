#ifndef CollinearFitAtTM_h_
#define CollinearFitAtTM_h_

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"

// #include "Workspace/TrajectoryMeasurementFits/interface/RandomVector.h"

/** Constrained fit at a TrajectoryMeasurement assuming collinearity 
 *  of incoming / outgoing momenta. The result of the fit is a vector
 *  of 6 variables: the first five correspond to local trajectory
 *  parameters for the incoming momentum, the 6th is the estimated 
 *  remaining energy fraction (p_out / p_in). The NDF are 6 (4)
 *  for a valid (invalid) RecHit. **/

class CollinearFitAtTM {
public:
  /// parameter indices in the result vector / covariance matrix
  enum { ParQpIn=0, ParQpOut, ParDxDz, ParDyDz, ParX, ParY };

  CollinearFitAtTM ();

  typedef ROOT::Math::SVector<double,6> ResultVector;
  typedef ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> > ResultMatrix;
  /// Fit for one TM. Return value "true" for success. 
  bool fit (const TrajectoryMeasurement& tm,
	    ResultVector& parameters,
	    ResultMatrix& covariance,
	    double& chi2);
  /// Fit with explicit input parameters. Return value "true" for success. 
  bool fit (const AlgebraicVector5& fwdParameters, 
	    const AlgebraicSymMatrix55& fwdCovariance,
	    const AlgebraicVector5& bwdParameters, 
	    const AlgebraicSymMatrix55& bwdCovariance,
	    const LocalPoint& hitPosition, const LocalError& hitErrors,
	    ResultVector& parameters,
	    ResultMatrix& covariance,
	    double& chi2);
private:
  ROOT::Math::SMatrix<double,12,6> jacobian_;
  ROOT::Math::SVector<double,12> measurements_;  
  ROOT::Math::SMatrix<double,12,12,ROOT::Math::MatRepSym<double,12> > weightMatrix_;
  ROOT::Math::SVector<double,6> projectedMeasurements_;
//   RandomVector randomGenerator;
};

#endif
