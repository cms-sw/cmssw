#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAnalFitAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAnalFitAlgo_HH

/** \class EcalUncalibRecHitRecAnalFitAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using an analytical fit
  *
  *  $Id: $
  *  $Date: 2005/10/25 09:10:01 $
  *  $Revision: 1.1 $
  *  \author A. Palma, Sh. Rahatlou Roma1
  */

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include <vector>

template<class C> class EcalUncalibRecHitRecAnalFitAlgo : public EcalUncalibRecHitRecAbsAlgo<C>
{
 public:
  // destructor
  virtual ~EcalUncalibRecHitRecAnalFitAlgo<C>() { };

  /// Compute parameters
  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, const std::vector<double>& pedestals,
                                            const std::vector<HepMatrix>& weights,
                                            const std::vector<HepSymMatrix>& chi2Matrix) {
    double amplitude_(-1.),  pedestal_(-1.), jitter_(-1.), chi2_(-1.);

    // Get time samples
    /*
    HepMatrix frame(C::MAXSAMPLES, 1);
    int gainId0 = dataFrame.sample(0).gainId();
    int iGainSwitch = 0;
    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
      frame[iSample][0] = double(dataFrame.sample(iSample).adc());
      if (dataFrame.sample(iSample).gainId() > gainId0) iGainSwitch = 1;
    }
    */

    // Compute parameters
    //std::cout << "EcalUncalibRecHitRecAnalFitAlgo::makeRecHit() not yey implemented. returning dummy rechit" << std::endl;

    return EcalUncalibratedRecHit( dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_);
  }
};
#endif
