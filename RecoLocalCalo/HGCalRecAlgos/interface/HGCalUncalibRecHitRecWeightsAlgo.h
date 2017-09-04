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

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

template<class C> class HGCalUncalibRecHitRecWeightsAlgo 
{
 public:
  // destructor
  virtual ~HGCalUncalibRecHitRecWeightsAlgo<C>() { };

  void set_isSiFESim(const bool isSiFE) { isSiFESim_ = isSiFE; }
  bool isSiFESim() const { return isSiFESim_; }

  void set_ADCLSB(const double adclsb) { adcLSB_ = adclsb; }
  void set_TDCLSB(const double tdclsb) { tdcLSB_ = tdclsb; }

  void set_toaLSBToNS(const double lsb2ns) { toaLSBToNS_ = lsb2ns; }

  void set_tdcOnsetfC(const double tdcOnset) { tdcOnsetfC_ = tdcOnset; }

  void set_fCPerMIP(const std::vector<double>& fCPerMIP) { 
    if( std::any_of(fCPerMIP.cbegin(), 
                    fCPerMIP.cend(), 
                    [](double conv){ return conv <= 0.0; }) ) {
      throw cms::Exception("BadConversionFactor") << "At least one of fCPerMIP is zero!" << std::endl;
    }
    fCPerMIP_ = fCPerMIP; 
  }
  
  void setGeometry(const HGCalGeometry* geom) { 
    if ( geom ) ddd_ = &(geom->topology().dddConstants());
    else ddd_ = nullptr;
  }

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
        // LG (11/04/2016):
        // offset the TDC upwards to reflect the bin center
	amplitude_ = ( std::floor(tdcOnsetfC_/adcLSB_) + 1.0 )* adcLSB_ + ( double(sample.data()) + 0.5) * tdcLSB_;
	if(sample.getToAValid()) jitter_    = double(sample.toa()) * toaLSBToNS_;
	LogDebug("HGCUncalibratedRecHit") << "TDC+: set the charge to: " << amplitude_ << ' ' << sample.data() 
                                          << ' ' << tdcLSB_ << std::endl
                                          << "TDC+: set the ToA to: " << jitter_ << ' ' 
                                          << sample.toa() << ' ' << toaLSBToNS_ << ' '
                                          << " flag=" << flag << std::endl;
      } else {
	amplitude_ = double(sample.data()) * adcLSB_;
	if(sample.getToAValid()) {jitter_    = double(sample.toa()) * toaLSBToNS_;}
	LogDebug("HGCUncalibratedRecHit") << "ADC+: set the charge to: " << amplitude_ << ' ' << sample.data() 
                                          << ' ' << adcLSB_ << ' ' 
					  << "TDC+: set the ToA to: " << jitter_ << ' '
                                          << sample.toa() << ' ' << toaLSBToNS_ << ' '<< std::endl;
      }
    } else {
      amplitude_ = double(sample.data()) * adcLSB_;
      LogDebug("HGCUncalibratedRecHit") << "ADC+: set the charge to: " << amplitude_ << ' ' << sample.data() 
						 << ' ' << adcLSB_ << ' ' << std::endl;
    }
    
    int thickness = 1;
    if( ddd_ != nullptr ) {
      HGCalDetId hid(dataFrame.id());
      thickness = ddd_->waferTypeL(hid.wafer());
    }    
    amplitude_ = amplitude_/fCPerMIP_[thickness-1];

    LogDebug("HGCUncalibratedRecHit") << "Final uncalibrated amplitude : " << amplitude_ << std::endl;
    
    return HGCUncalibratedRecHit( dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_, flag);
   }
  
 private:
   double adcLSB_, tdcLSB_, toaLSBToNS_, tdcOnsetfC_;
   bool   isSiFESim_;  
   std::vector<double> fCPerMIP_;
   std::array<float, 3> tdcForToaOnsetfC_;
   const HGCalDDDConstants* ddd_;
};
#endif
