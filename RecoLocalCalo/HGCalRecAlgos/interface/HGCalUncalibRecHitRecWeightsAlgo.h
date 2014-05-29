#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalUncalibRecHitRecWeightsAlgo_HH
#define RecoLocalCalo_HGCalRecAlgos_HGCalUncalibRecHitRecWeightsAlgo_HH

/** \class HGalUncalibRecHitRecWeightsAlgo
  *  compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method, a la Ecal 
  *
  *  \author Valeri Andreev
  *  
  *
  */

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalUncalibRecHitRecAbsAlgo.h"
#include "Math/SVector.h"
#include <vector>

template<class C> class HGCalUncalibRecHitRecWeightsAlgo 
{
 public:
  // destructor
  virtual ~HGCalUncalibRecHitRecWeightsAlgo<C>() { };

  /// Compute parameters
   virtual HGCUncalibratedRecHit makeRecHit(
					      const C& dataFrame 
    ) {
    double amplitude_(-1.),  pedestal_(-1.), jitter_(-1.), chi2_(-1.);
    uint32_t flag = 0;
    double energy = 0;

    //    static const int MAXSAMPLES = 10;
    //    ROOT::Math::SVector<double,MAXSAMPLES> frame;

    for (int iSample = 0 ; iSample < dataFrame.size(); ++iSample)
	    {
	      //	      frame(iSample) = double(dataFrame.sample(iSample).adc());
	      energy += double(dataFrame.sample(iSample).adc());
	    }

    amplitude_ = energy; // fast-track simhits propagation
    
    return HGCUncalibratedRecHit( dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_, flag);
   }
};
#endif
