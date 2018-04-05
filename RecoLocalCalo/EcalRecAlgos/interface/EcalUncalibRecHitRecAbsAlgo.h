#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAbsAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAbsAlgo_HH

/** \class EcalUncalibRecHitRecAbsAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
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
  virtual ~EcalUncalibRecHitRecAbsAlgo() = default;

  /// make rechits from dataframes

  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, 
					    const double* pedestals,
					    const double* gainRatios,
					    const EcalWeightSet::EcalWeightMatrix** weights, 
					    const EcalWeightSet::EcalChi2WeightMatrix** chi2Matrix) = 0;

};
#endif
