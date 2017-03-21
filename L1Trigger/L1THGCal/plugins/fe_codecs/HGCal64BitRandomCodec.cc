#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCal64BitRandomCodec.h"

using namespace HGCalTriggerFE;

DEFINE_EDM_PLUGIN(HGCalTriggerFECodecFactory, 
                  HGCal64BitRandomCodec,
                  "HGCal64BitRandomCodec");

void HGCal64BitRandomCodec::
setDataPayloadImpl(const HGCEEDigiCollection&,
                   const HGCHEDigiCollection&,
                   const HGCHEDigiCollection& ) {
  codecImpl_.setDataPayload(data_);
}

void HGCal64BitRandomCodec::
setDataPayloadImpl(const l1t::HGCFETriggerDigi& digi) {
  codecImpl_.setDataPayload(data_);
}

std::vector<bool>
HGCal64BitRandomCodec::
encodeImpl(const HGCal64BitRandomCodec::data_type& data) const {
  return codecImpl_.encode(data);
}

HGCal64BitRandomCodec::data_type
HGCal64BitRandomCodec::
decodeImpl(const std::vector<bool>& data, const uint32_t) const {
  return codecImpl_.decode(data);
}


