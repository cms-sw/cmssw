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
#include <vector>

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

template<class C> class HGCalUncalibRecHitRecWeightsAlgo 
{
 public:
  // destructor
  virtual ~HGCalUncalibRecHitRecWeightsAlgo<C>() { };

  virtual void set_isSiFESim(const bool isSiFE) { isSiFESim_ = isSiFE; }

  virtual void set_ADCLSBInfC(const double adclsb) { adcLSBInfC_ = adclsb; }
  virtual void set_TDCLSBInfC(const double tdclsb) { tdcLSBInfC_ = tdclsb; }

  virtual void set_fCToMIP(const double fc2mip) { fCToMIP_ = fc2mip; }
  virtual void set_ADCToMIP(const double adc2mip) { adcToMIP_ = adc2mip; }
  
  /// Compute parameters
  virtual HGCUncalibratedRecHit makeRecHit( const C& dataFrame ) {
    
    double amplitude_(-1.),  pedestal_(-1.), jitter_(-1.), chi2_(-1.);
    uint32_t flag = 0;
    double energy = 0;
    
    for (int iSample = 0 ; iSample < dataFrame.size(); ++iSample) {
      const auto& sample = dataFrame.sample(iSample);
      
      // are we using the SiFE Simulation?
      if( isSiFESim_ ) {
        // mode == true means we are running in ADC + TDC mode
        if( sample.mode() ) {
          // threshold == true is TDC, == false is ADC
          if( sample.threshold() ) energy += double(sample.data()) * tdcLSBInfC_ * fCToMIP_;
          else                     energy += double(sample.data()) * adcLSBInfC_ * fCToMIP_;
        } else { // false means we are ADC only mode
          energy += double(sample.data()) * adcLSBInfC_ * fCToMIP_;
        }              
      } else { // use adcToMIP value
        energy += double(sample.data()) * adcToMIP_;
      }
    }
    
    amplitude_ = energy; // fast-track simhits propagation
    
    return HGCUncalibratedRecHit( dataFrame.id(), amplitude_, 
                                  pedestal_, jitter_, chi2_, flag);
   }

 private:
   bool   isSiFESim_;
   double adcLSBInfC_, tdcLSBInfC_, fCToMIP_, adcToMIP_;
};
#endif
