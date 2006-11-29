#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecWeightsAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecWeightsAlgo_HH

/** \class EcalUncalibRecHitRecWeightsAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  $Id: EcalUncalibRecHitRecWeightsAlgo.h,v 1.2 2006/06/19 18:51:53 meridian Exp $
  *  $Date: 2006/06/19 18:51:53 $
  *  $Revision: 1.2 $
  *  \author R. Bruneliere - A. Zabi
  */

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include <vector>

template<class C> class EcalUncalibRecHitRecWeightsAlgo : public EcalUncalibRecHitRecAbsAlgo<C>
{
 public:
  // destructor
  virtual ~EcalUncalibRecHitRecWeightsAlgo<C>() { };

  /// Compute parameters
  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, const std::vector<double>& pedestals,
					    const std::vector<double>& gainRatios,
					    const std::vector<HepMatrix>& weights,
					    const std::vector<HepSymMatrix>& chi2Matrix) {
    double amplitude_(-1.),  pedestal_(-1.), jitter_(-1.), chi2_(-1.);

    // Get time samples
    HepMatrix frame(C::MAXSAMPLES, 1);
    int gainId0 = 1;
    int iGainSwitch = 0;
    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
      int gainId = dataFrame.sample(iSample).gainId(); 
      if (gainId != gainId0) iGainSwitch = 1;
      if (!iGainSwitch)
	frame[iSample][0] = double(dataFrame.sample(iSample).adc());
      else
	frame[iSample][0] = double(((double)(dataFrame.sample(iSample).adc()) - pedestals[gainId-1]) * gainRatios[gainId-1]);
    }

    // Compute parameters
    HepMatrix param = weights[iGainSwitch]*frame;
    amplitude_ = param[EcalUncalibRecHitRecAbsAlgo<C>::iAmplitude][0];
    pedestal_ = param[EcalUncalibRecHitRecAbsAlgo<C>::iPedestal][0];
    if (amplitude_) jitter_ = param[EcalUncalibRecHitRecAbsAlgo<C>::iTime][0]/amplitude_;

    // Compute chi2 = frame^T * chi2Matrix * frame
    HepMatrix tFrame = frame.T();
    HepMatrix tFrameChi2 = tFrame*chi2Matrix[iGainSwitch];
    chi2_ = (tFrameChi2*frame)[0][0];

    return EcalUncalibratedRecHit( dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_);
  }
};
#endif
