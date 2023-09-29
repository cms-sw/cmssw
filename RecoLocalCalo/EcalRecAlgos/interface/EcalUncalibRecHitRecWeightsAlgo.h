#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecWeightsAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecWeightsAlgo_HH

/** \class EcalUncalibRecHitRecWeightsAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  \author R. Bruneliere - A. Zabi
  *  
  *  The chi2 computation with matrix is replaced by the chi2express which is  moved outside the weight algo
  *  (need to clean up the interface in next iteration so that we do not pass-by useless arrays)
  *
  */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"

#include <vector>

template <class C>
class EcalUncalibRecHitRecWeightsAlgo {
public:
  // destructor
  virtual ~EcalUncalibRecHitRecWeightsAlgo(){};

  /// Compute parameters
  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame,
                                            const double* pedestals,
                                            const double* pedestalsRMS,
                                            const double* gainRatios,
                                            const EcalWeightSet::EcalWeightMatrix** weights,
                                            const EcalShapeBase& testbeamPulseShape) {
    double amplitude_(-1.), pedestal_(-1.), jitter_(-1.), chi2_(-1.);
    uint32_t flag = 0;

    // Get time samples
    ROOT::Math::SVector<double, C::MAXSAMPLES> frame;
    int gainId0 = 1;
    int iGainSwitch = 0;
    bool isSaturated = false;
    for (int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
      int gainId = dataFrame.sample(iSample).gainId();
      //Handling saturation (treating saturated gainId as maximum gain)
      if (gainId == 0) {
        gainId = 3;
        //isSaturated = 1;
        // in pileup run May2012 samples 7,8,9,10 have gainid ==0
        // fix it like this: it won't hurt for the future SA20120512
        if (iSample == 4 || iSample == 5 || iSample == 6)
          isSaturated = true;
      }

      //      if (gainId != gainId0) iGainSwitch = 1;
      // same problem as above: mark saturation only when physically
      // expected to occur SA20120513
      if ((gainId != gainId0) && (iSample == 4 || iSample == 5 || iSample == 6))
        iGainSwitch = 1;
      if (!iGainSwitch)
        frame(iSample) = double(dataFrame.sample(iSample).adc());
      else
        frame(iSample) =
            double(((double)(dataFrame.sample(iSample).adc()) - pedestals[gainId - 1]) * gainRatios[gainId - 1]);
    }

    // Compute parameters
    ROOT::Math::SVector<double, 3> param = (*(weights[iGainSwitch])) * frame;
    amplitude_ = param(EcalUncalibRecHitRecAbsAlgo<C>::iAmplitude);
    pedestal_ = param(EcalUncalibRecHitRecAbsAlgo<C>::iPedestal);
    if (amplitude_)
      jitter_ = -param(EcalUncalibRecHitRecAbsAlgo<C>::iTime) / amplitude_;
    else
      jitter_ = 0.;

    //When saturated gain flag i
    if (isSaturated) {
      flag = EcalUncalibratedRecHit::kSaturated;
      amplitude_ = double((4095. - pedestals[2]) * gainRatios[2]);
    }
    return EcalUncalibratedRecHit(dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_, flag);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};
#endif
