#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAbsAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAbsAlgo_HH

/** \class EcalUncalibRecHitRecAbsAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  $Id: EcalUncalibRecHitRecAbsAlgo.h,v 1.2 2006/06/19 18:51:53 meridian Exp $
  *  $Date: 2006/06/19 18:51:53 $
  *  $Revision: 1.2 $
  *  \author R. Bruneliere - A. Zabi
  */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include <vector>
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"

template<class C> class EcalUncalibRecHitRecAbsAlgo
{
 public:
  enum { nWeightsRows = 3, iAmplitude = 0, iPedestal = 1, iTime = 2 };

  /// Constructor
  //EcalUncalibRecHitRecAbsAlgo() { };

  /// Destructor
  //virtual ~EcalUncalibRecHitRecAbsAlgo() { };

  /// make rechits from dataframes

  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, 
					    const double* pedestals,
					    const double* gainRatios,
					    const math::EcalWeightMatrix::type** weights, 
					    const math::EcalChi2WeightMatrix::type** chi2Matrix) = 0;

};
#endif
