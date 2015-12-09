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
#include <cmath>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

template<class C> class HGCalUncalibRecHitRecWeightsAlgo 
{
 public:
  // destructor
  virtual ~HGCalUncalibRecHitRecWeightsAlgo<C>() { };

  virtual void set_isSiFESim(const bool isSiFE) { isSiFESim_ = isSiFE; }

  virtual void set_ADCLSB(const double adclsb) { adcLSB_ = adclsb; }
  virtual void set_TDCLSB(const double tdclsb) { tdcLSB_ = tdclsb; }

  virtual void set_toaLSBToNS(const double lsb2ns) { toaLSBToNS_ = lsb2ns; }

  virtual void set_tdcOnsetfC(const double tdcOnset) { tdcOnsetfC_ = tdcOnset; }

  /// Compute parameters
  virtual HGCUncalibratedRecHit makeRecHit( const C& dataFrame ) {
    double amplitude_(-1.),  pedestal_(-1.), jitter_(-1.), chi2_(-1.);
    uint32_t flag = 0;
    
    constexpr int iSample=2; //only in-time sample
    const auto& sample = dataFrame.sample(iSample);
    
    // are we using the SiFE Simulation?
    if( isSiFESim_ ) {
      // mode == true means TDC readout was activated
      if( sample.mode() ) {
	flag       = !sample.threshold();  //raise flag if busy cell
        // LG (23/06/2015): 
        //to get a continuous energy spectrum we must add here the maximum value in fC ever
        //reported by the ADC. Namely: floor(tdcOnset/adcLSB_) * adcLSB_
        // need to increment by one so TDC doesn't overlap with ADC last bin
	amplitude_ = ( std::floor(tdcOnsetfC_/adcLSB_) + 1.0 )* adcLSB_ + double(sample.data()) * tdcLSB_;
	jitter_    = double(sample.toa()) * toaLSBToNS_;
	LogDebug("HGCUncalibratedRecHit") << "TDC+: set the energy to: " << amplitude_ << ' ' << sample.data() 
                                          << ' ' << tdcLSB_ << std::endl
                                          << "TDC+: set the jitter to: " << jitter_ << ' ' 
                                          << sample.toa() << ' ' << toaLSBToNS_ << ' '
                                          << " flag=" << flag << std::endl;
      } 
      else {
	amplitude_ = double(sample.data()) * adcLSB_;
	LogDebug("HGCUncalibratedRecHit") << "ADC+: set the energy to: " << amplitude_ << ' ' << sample.data() 
                                          << ' ' << adcLSB_ << ' ' << std::endl;
      }
    }
    else {
      amplitude_ = double(sample.data()) * adcLSB_;
      LogDebug("HGCUncalibratedRecHit") << "ADC+: set the energy to: " << amplitude_ << ' ' << sample.data() 
                                        << ' ' << adcLSB_ << ' ' << std::endl;
    }
    
    LogDebug("HGCUncalibratedRecHit") << "Final uncalibrated amplitude : " << amplitude_ << std::endl;
    return HGCUncalibratedRecHit( dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_, flag);
   }
  
 private:
   double adcLSB_, tdcLSB_, fCToMIP_, toaLSBToNS_, tdcOnsetfC_;
   bool   isSiFESim_;
};
#endif
