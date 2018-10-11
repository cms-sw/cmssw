#ifndef __L1Trigger_L1THGCal_HGCalVFECompressionImpl_h__
#define __L1Trigger_L1THGCal_HGCalVFECompressionImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

class HGCalVFECompressionImpl
{

 public:
  HGCalVFECompressionImpl(const edm::ParameterSet& conf);

  void compress(const std::map<HGCalDetId, uint32_t>&,
                std::map<HGCalDetId, std::array<uint32_t, 2> >&);
  uint8_t compressSingle(const uint32_t value);
  uint32_t decompressSingle(const uint8_t code);
     
 private:
  uint32_t exponentBits_;
  uint32_t mantissaBits_;
  bool     rounding_;
  bool     saturable_;
  uint32_t saturationValue_;
  uint32_t compressedValueLUT_[256];
     
};

#endif
