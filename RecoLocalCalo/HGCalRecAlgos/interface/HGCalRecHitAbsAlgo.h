#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalRecHitAbsAlgo_HH
#define RecoLocalCalo_HGCalRecAlgos_HGCalRecHitAbsAlgo_HH

/** \class HGCalRecHitAbsAlgo
  *  Template algorithm to make rechits from uncalibrated rechits
  *
  *
  *
  *  \author Valeri Andreev
  */

#include <vector>
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCUncalibratedRecHit.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

class HGCalRecHitAbsAlgo {
public:
  /// Constructor
  //HGCalRecHitAbsAlgo() { };

  /// Destructor
  virtual ~HGCalRecHitAbsAlgo(){};

  inline void set(const CaloGeometry& geom) { rhtools_.setGeometry(geom); }

  /// make rechits from dataframes
  virtual void setLayerWeights(const std::vector<float>& weights){};

  virtual void setADCToGeVConstant(const float value) = 0;
  virtual HGCRecHit makeRecHit(const HGCUncalibratedRecHit& uncalibRH, const uint32_t& flags) const = 0;

protected:
  hgcal::RecHitTools rhtools_;
};
#endif
