#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalUncalibRecHitRecAbsAlgo_HH
#define RecoLocalCalo_HGCalRecAlgos_HGCalUncalibRecHitRecAbsAlgo_HH

/** \class HGCalUncalibRecHitRecAbsAlgo
  *  Template used by Ecal to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  \author
  */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include <vector>
#include "DataFormats/HGCRecHit/interface/HGCUncalibratedRecHit.h"

template<class C> class HGCalUncalibRecHitRecAbsAlgo
{
 public:
  enum { nWeightsRows = 3, iAmplitude = 0, iPedestal = 1, iTime = 2 };  // in the case of saturation

  /// Constructor

  virtual HGCUncalibratedRecHit makeRecHit(const C& dataFrame 
					    ) = 0;

};
#endif
