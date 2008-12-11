#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecWeightsAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecWeightsAlgo_HH

/** \class EcalUncalibRecHitRecWeightsAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  $Id: EcalUncalibRecHitRecWeightsAlgo.h,v 1.5 2007/04/05 15:41:21 meridian Exp $
  *  $Date: 2007/04/05 15:41:21 $
  *  $Revision: 1.5 $
  *  \author R. Bruneliere - A. Zabi
  */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include <vector>

template<class C> class EcalUncalibRecHitRecWeightsAlgo : public EcalUncalibRecHitRecAbsAlgo<C>
{
 public:
  // destructor
  virtual ~EcalUncalibRecHitRecWeightsAlgo<C>() { };

  /// Compute parameters
  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, const double* pedestals,
					    const double* gainRatios,
					    const EcalWeightSet::EcalWeightMatrix** weights, 
					    const EcalWeightSet::EcalChi2WeightMatrix** chi2Matrix) {
    double amplitude_(-1.),  pedestal_(-1.), jitter_(-1.), chi2_(-1.);

    // Get time samples
    ROOT::Math::SVector<double,C::MAXSAMPLES> frame;
    int gainId0 = 1;
    int iGainSwitch = 0;
    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
      int gainId = dataFrame.sample(iSample).gainId();
      //Handling saturation (treating saturated gainId as maximum gain)
      if (gainId == 0 ) gainId = 3;
      if (gainId != gainId0) iGainSwitch = 1;
      if (!iGainSwitch)
	frame(iSample) = double(dataFrame.sample(iSample).adc());
      else
	frame(iSample) = double(((double)(dataFrame.sample(iSample).adc()) - pedestals[gainId-1]) * gainRatios[gainId-1]);
    }

    // Compute parameters
    ROOT::Math::SVector <double,3> param = (*(weights[iGainSwitch])) * frame;
    amplitude_ = param(EcalUncalibRecHitRecAbsAlgo<C>::iAmplitude);
    pedestal_ = param(EcalUncalibRecHitRecAbsAlgo<C>::iPedestal);
    if (amplitude_) jitter_ = param(EcalUncalibRecHitRecAbsAlgo<C>::iTime);
    // Compute chi2 = frame^T * chi2Matrix * frame
    chi2_ = ROOT::Math::Similarity((*(chi2Matrix[iGainSwitch])),frame);
    return EcalUncalibratedRecHit( dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_);
  }
};
#endif
