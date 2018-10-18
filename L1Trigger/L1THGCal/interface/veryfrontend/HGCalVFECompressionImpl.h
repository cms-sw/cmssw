#ifndef __L1Trigger_L1THGCal_HGCalVFECompressionImpl_h__
#define __L1Trigger_L1THGCal_HGCalVFECompressionImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

class HGCalVFECompressionImpl
{

 public:
  HGCalVFECompressionImpl(const edm::ParameterSet& conf);

  uint32_t bitLength(uint32_t x);
  void compress(const std::map<HGCalDetId, uint32_t>&,
                std::map<HGCalDetId, std::array<uint32_t, 2> >&);
  uint32_t compressSingle(const uint32_t value);
  uint32_t decompressSingle(const uint32_t code);
     
 private:
  uint32_t exponentBits_;
  uint32_t mantissaBits_;
  bool     rounding_;
  bool     saturable_;
  uint32_t saturationCode_;
  uint32_t saturationValue_;
     
};

#endif
