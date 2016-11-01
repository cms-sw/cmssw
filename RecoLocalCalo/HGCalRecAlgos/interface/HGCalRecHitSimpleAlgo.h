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
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <iostream>

class HGCalRecHitSimpleAlgo : public HGCalRecHitAbsAlgo {
 public:
  // default ctor
  HGCalRecHitSimpleAlgo() {
    adcToGeVConstant_ = -1;
    adcToGeVConstantIsSet_ = false;
  }

  virtual void setLayerWeights(const std::vector<float>& weights) override {
    weights_ = weights;
  }
  
  virtual void setADCToGeVConstant(const float value) override {
    adcToGeVConstant_ = value;
    adcToGeVConstantIsSet_ = true;
  }
  
  
  // destructor
  virtual ~HGCalRecHitSimpleAlgo() { };
  
  /// Compute parameters
  virtual HGCRecHit makeRecHit(const HGCUncalibratedRecHit& uncalibRH,
                               const uint32_t& flags = 0) const override {
    
    if(!adcToGeVConstantIsSet_) {
      throw cms::Exception("HGCalRecHitSimpleAlgoBadConfig") 
        << "makeRecHit: adcToGeVConstant_ not set before calling this method!";
    }
    
    DetId baseid = uncalibRH.id();
    unsigned layer = 0;
    if( DetId::Hcal == baseid.det() && HcalEndcap == baseid.subdetId() ) {
      layer =  HcalDetId(baseid).depth() + 40;
    } else if ( DetId::Forward == baseid.det() && HGCEE == baseid.subdetId() ) {
      layer = HGCalDetId(baseid).layer();
    }else if ( DetId::Forward == baseid.det() && HGCHEF == baseid.subdetId() ) {
      layer = HGCalDetId(baseid).layer() + 28;
    } else {
      throw cms::Exception("InvalidRecHit")
	<< "HGCalRecHitSimpleAlgo encountered a non-HGCal det id: " << baseid.det() << ' ' << baseid.subdetId() << ' ' << baseid.rawId();
    }

    HGCalDetId hid(uncalibRH.id());


    //    float clockToNsConstant = 25;    
    float energy = uncalibRH.amplitude() * weights_[layer] * 0.001f;
    float time   = uncalibRH.jitter();

    //if(time<0) time   = 0; // fast-track digi conversion
    
    HGCRecHit rh( uncalibRH.id(), energy, time );
    
    // Now fill flags
    // all rechits from the digitizer are "good" at present
    rh.setFlag(HGCRecHit::kGood);
    
    return rh;
  }

private:
  float adcToGeVConstant_;
  bool  adcToGeVConstantIsSet_;
  std::vector<float> weights_;
};
#endif
