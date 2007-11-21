#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitMaxSampleAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitMaxSampleAlgo_HH

/** \class EcalUncalibRecHitMaxSampleAlgo
  *  Amplitude reconstucted by the difference MAX_adc - min_adc
  *  jitter is sample number of MAX_adc, pedestal is min_adc
  *
  *  \author G. Franzoni, E. Di Marco
  */

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


template<class C> class EcalUncalibRecHitMaxSampleAlgo : public EcalUncalibRecHitRecAbsAlgo<C>
{
  
 public:
  
  virtual ~EcalUncalibRecHitMaxSampleAlgo<C>() { };
  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, const double* pedestals,
					    const double* gainRatios,
					    const EcalWeightSet::EcalWeightMatrix** weights,
                                            const EcalWeightSet::EcalChi2WeightMatrix** chi2Matrix);
};

/// compute rechits
template<class C> EcalUncalibratedRecHit  
EcalUncalibRecHitMaxSampleAlgo<C>::makeRecHit(const C& dataFrame, const double* pedestals,
					      const double* gainRatios,
					      const EcalWeightSet::EcalWeightMatrix** weights,
					      const EcalWeightSet::EcalChi2WeightMatrix** chi2Matrix) {
  
  double amplitude_(-1.),  pedestal_(4095.), jitter_(-1.), chi2_(-1.);
  
  int gainId=-1;
  double sampleAdc;

  for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
    
    gainId = dataFrame.sample(iSample).gainId(); 
	
    // ampli gain 12
    if ( gainId == 1){
      sampleAdc = dataFrame.sample(iSample).adc();
    }
      
    else
      {
	if ( gainId == 2){ 	  // ampli gain 6
	  sampleAdc = 200 + (dataFrame.sample(iSample).adc() - 200) * 2 ;
	}
	else  {  // accounts for gainId==3 or 0 - ampli gain 1 and gain0
	  sampleAdc = 200 + (dataFrame.sample(iSample).adc() - 200) * 12 ;
	}
      }
    
    if( sampleAdc >amplitude_ )	  {
      amplitude_ = sampleAdc;
      jitter_ = iSample;
    }// if statement
    
    if (sampleAdc<pedestal_) pedestal_ = sampleAdc;

  }// loop on samples
      
      
  return EcalUncalibratedRecHit( dataFrame.id(), amplitude_-pedestal_ , pedestal_, jitter_, chi2_);
}

#endif
