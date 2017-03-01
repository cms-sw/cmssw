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

class HGCalRecHitAbsAlgo
{
 public:

  /// Constructor
  //HGCalRecHitAbsAlgo() { };

  /// Destructor
  virtual ~HGCalRecHitAbsAlgo() { };

  /// make rechits from dataframes
  virtual void setLayerWeights(const std::vector<float>& weights) {};

  virtual void setADCToGeVConstant(const float value) = 0;
  virtual HGCRecHit makeRecHit(const HGCUncalibratedRecHit& uncalibRH, const uint32_t &flags) const = 0;

};
#endif
