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
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

template <class C>
class HGCalUncalibRecHitRecWeightsAlgo {
public:
  // destructor
  virtual ~HGCalUncalibRecHitRecWeightsAlgo<C>(){};

  void set_isSiFESim(const bool isSiFE) { isSiFESim_ = isSiFE; }
  bool isSiFESim() const { return isSiFESim_; }

  void set_ADCLSB(const double adclsb) { adcLSB_ = adclsb; }
  void set_TDCLSB(const double tdclsb) { tdcLSB_ = tdclsb; }

  void set_toaLSBToNS(const double lsb2ns) { toaLSBToNS_ = lsb2ns; }

  void set_tdcOnsetfC(const double tdcOnset) { tdcOnsetfC_ = tdcOnset; }

  void set_fCPerMIP(const std::vector<double>& fCPerMIP) {
    if (std::any_of(fCPerMIP.cbegin(), fCPerMIP.cend(), [](double conv) { return conv <= 0.0; })) {
      throw cms::Exception("BadConversionFactor") << "At least one of fCPerMIP is zero!" << std::endl;
    }
    fCPerMIP_ = fCPerMIP;
  }

  void setGeometry(const HGCalGeometry* geom) {
    if (geom)
      ddd_ = &(geom->topology().dddConstants());
    else
      ddd_ = nullptr;
  }

  /// Compute HGCUncalibratedRecHitTurn from DataFrame
  virtual HGCUncalibratedRecHit makeRecHit(const C& dataFrame) {

    double amplitude_(-1.), pedestal_(-1.), jitter_(-1.), chi2_(-1.);
    uint32_t flag = 0;

    constexpr int iSample = 2;  //only in-time sample
    const auto& sample = dataFrame.sample(iSample);

    // Have digis been done with the complete digitization with the signal shape?
    // (originally done only for the silicon, while for scitillator it was trivial)
    if (isSiFESim_) {

      // mode == true : TDC readout was activated and amplitude comes from TimeOverThreshold
      if (sample.mode())
	{
	  std::cout << "isSiFESim mode" << std::endl; // GF
	  flag = !sample.threshold();  // raise flag if busy cell
	  // LG (23/06/2015):
	  // to get a continuous energy spectrum we must add here the maximum value in fC ever
	  // reported by the ADC. Namely: floor(tdcOnset/adcLSB_) * adcLSB_
	  // need to increment by one so TDC doesn't overlap with ADC last bin
	  // LG (11/04/2016):
	  // offset the TDC upwards to reflect the bin center
	  amplitude_ = (std::floor(tdcOnsetfC_ / adcLSB_) + 1.0) * adcLSB_ + (double(sample.data()) + 0.5) * tdcLSB_;
	  
	  if (sample.getToAValid())  {
	    jitter_ = double(sample.toa()) * toaLSBToNS_; }
//	  LogDebug("HGCUncalibratedRecHit")
//	    << "isSiFESim_: TDC+: set the charge to: " << amplitude_ << ' ' << sample.data() << ' ' << tdcLSB_ << std::endl
//	    << "            TDC+: set the ToA to: " << jitter_ << ' ' << sample.toa() << ' ' << toaLSBToNS_ << ' '
//	    << "            flag=" << flag << std::endl;
	}
      else 
	{
	  std::cout << "isSiFESim not mode" << std::endl;
	  amplitude_ = double(sample.data()) * adcLSB_; // why do we not have +0.5 here ?
	  if (sample.getToAValid()) {
	    jitter_ = double(sample.toa()) * toaLSBToNS_; }
//	  LogDebug("HGCUncalibratedRecHit")
//	    << "isSiFESim_: " << isSiFESim_ << " ADC+: set the charge to: " << amplitude_ << ' ' << sample.data() << ' ' << adcLSB_ << ' '
//	    << "               TDC+: set the ToA to: " << jitter_ << ' ' << sample.toa() << ' ' << toaLSBToNS_ << ' ' << tdcLSB_
//	    << "               getToAValid(): " << sample.getToAValid() << " mode(): " << sample.mode()	    
//	    << std::endl;
	}
    }

    // trivial digitization, i.e. no signal shape
    else 
      {
	//std::cout << "not isSiFESim" << std::endl;
	amplitude_ = double(sample.data()) * adcLSB_;
	//LogDebug("HGCUncalibratedRecHit") << "ADC+: set the charge to: " << amplitude_ << ' ' << sample.data() << ' '
	//				  << adcLSB_ << ' ' << std::endl;
      }
    
    int thickness = (ddd_ != nullptr) ? ddd_->waferType(dataFrame.id()) : 0;
    amplitude_ = amplitude_ / fCPerMIP_[thickness];
    
    LogDebug("HGCUncalibratedRecHit")
      << "isSiFESim_: " << isSiFESim_ << " ADC+: set the charge to: " << amplitude_ << ' ' << sample.data() << ' ' << adcLSB_ << ' '
      << "               TDC+: set the ToA to: " << jitter_ << ' ' << sample.toa() << ' ' << toaLSBToNS_ << ' ' << tdcLSB_
      << "               getToAValid(): " << sample.getToAValid() << " mode(): " << sample.mode()	    
      << std::endl;
    
    LogDebug("HGCUncalibratedRecHit") << "Final uncalibrated amplitude : " << amplitude_ << std::endl;
    
    //std::cout << "detID: 0x" << dataFrame.id().rawId() << " ampli: " << amplitude_ 
    // << " jitter : "  << jitter_ << " samplep[2]: " << sample.data() << std::endl;
    return HGCUncalibratedRecHit(dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_, flag);
  }
  
 private:
  double adcLSB_, tdcLSB_, toaLSBToNS_, tdcOnsetfC_;
  bool isSiFESim_;
  std::vector<double> fCPerMIP_;
  const HGCalDDDConstants* ddd_;
};
#endif
