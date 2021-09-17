#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalRecHitSimpleAlgo_HH
#define RecoLocalCalo_HGCalRecAlgos_HGCalRecHitSimpleAlgo_HH

/** \class HGCalRecHitSimpleAlgo
  *  Simple algoritm to make HGCAL rechits from HGCAL uncalibrated rechits
  *  , following Ecal sceleton
  *
  *  \author Valeri Andreev
  */

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalRecHitAbsAlgo.h"
#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <iostream>

class HGCalRecHitSimpleAlgo : public HGCalRecHitAbsAlgo {
public:
  // default ctor
  HGCalRecHitSimpleAlgo() {
    adcToGeVConstant_ = -1;
    adcToGeVConstantIsSet_ = false;
  }

  void setLayerWeights(const std::vector<float>& weights) override { weights_ = weights; }

  void setNoseLayerWeights(const std::vector<float>& weights) { weightsNose_ = weights; }

  void setADCToGeVConstant(const float value) override {
    adcToGeVConstant_ = value;
    adcToGeVConstantIsSet_ = true;
  }

  // destructor
  ~HGCalRecHitSimpleAlgo() override{};

  /// Compute parameters
  HGCRecHit makeRecHit(const HGCUncalibratedRecHit& uncalibRH, const uint32_t& flags = 0) const override {
    if (!adcToGeVConstantIsSet_) {
      throw cms::Exception("HGCalRecHitSimpleAlgoBadConfig")
          << "makeRecHit: adcToGeVConstant_ not set before calling this method!";
    }

    DetId baseid = uncalibRH.id();
    unsigned layer = rhtools_.getLayerWithOffset(baseid);
    bool hfnose = DetId::Forward == baseid.det() && HFNose == baseid.subdetId();

    //    float clockToNsConstant = 25;
    float energy = (hfnose ? (uncalibRH.amplitude() * weightsNose_[layer] * 0.001f)
                           : (uncalibRH.amplitude() * weights_[layer] * 0.001f));
    float time = uncalibRH.jitter();

    //if(time<0) time   = 0; // fast-track digi conversion

    HGCRecHit rh(uncalibRH.id(), energy, time);

    // Now fill flags
    // all rechits from the digitizer are "good" at present
    rh.setFlag(HGCRecHit::kGood);

    return rh;
  }

private:
  float adcToGeVConstant_;
  bool adcToGeVConstantIsSet_;
  std::vector<float> weights_, weightsNose_;
};
#endif
