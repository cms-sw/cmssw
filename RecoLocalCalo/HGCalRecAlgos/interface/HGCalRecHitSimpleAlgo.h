#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalRecHitSimpleAlgo_HH
#define RecoLocalCalo_HGCalRecAlgos_HGCalRecHitSimpleAlgo_HH

/** \class HGCalRecHitSimpleAlgo
  *  Simple algoritm to make HGCAL rechits from HGCAL uncalibrated rechits
  *  , following Ecal sceleton
  *
  *  \author Valeri Andreev
  */

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalRecHitAbsAlgo.h"
#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"
#include "TMath.h"
#include <iostream>

class HGCalRecHitSimpleAlgo : public HGCalRecHitAbsAlgo {
 public:
  // default ctor
  HGCalRecHitSimpleAlgo() {
    adcToGeVConstant_ = -1;
    adcToGeVConstantIsSet_ = false;
  }

  virtual void setADCToGeVConstant(const float& value) {
    adcToGeVConstant_ = value;
    adcToGeVConstantIsSet_ = true;
  }


  // destructor
  virtual ~HGCalRecHitSimpleAlgo() { };

  /// Compute parameters
  virtual HGCRecHit makeRecHit(const HGCUncalibratedRecHit& uncalibRH,
			       //                                const float& intercalibConstant,
			       //                                const float& timeIntercalib = 0,
                                const uint32_t& flags = 0) const {

    if(!adcToGeVConstantIsSet_) {
      std::cout << "HGCalRecHitSimpleAlgo::makeRecHit: adcToGeVConstant_ not set before calling this method!" << 
                   " will use -1 and produce bogus rechits!" << std::endl;
    }

    //    float clockToNsConstant = 25;
    float energy = uncalibRH.amplitude() * adcToGeVConstant_;
    float time   = uncalibRH.jitter();
    if(time<0) time   = 0; // fast-track digi conversion

    HGCRecHit rh( uncalibRH.id(), energy, time );

    // Now fill flags

    bool good=true;
    

    if (good) rh.setFlag(HGCRecHit::kGood);
    return rh;
  }

private:
  float adcToGeVConstant_;
  bool  adcToGeVConstantIsSet_;

};
#endif
