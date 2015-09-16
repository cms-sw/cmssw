#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCal64BitRandomCodec.h"
#include <limits>

#include "TRandom3.h"

using namespace HGCalTriggerFE;

DEFINE_EDM_PLUGIN(HGCalTriggerFECodecFactory, 
                  HGCal64BitRandomCodec,
                  "HGCal64BitRandomCodec");

void HGCal64BitRandomCodec::
setDataPayloadImpl(const Module& , 
                   const HGCalTriggerGeometryBase& ,
                   const HGCEEDigiCollection&,
                   const HGCHEDigiCollection&,
                   const HGCHEDigiCollection& ) {
  data_.payload = 0;
  for( unsigned i = 0; i < 8*sizeof(data_type); ++i ) {
    data_.payload |= static_cast<uint64_t>(rand.Rndm() > 0.5) << i;
  }
}

std::vector<bool>
HGCal64BitRandomCodec::
encodeImpl(const HGCal64BitRandomCodec::data_type& data) const {
  std::vector<bool> result;
  result.resize(8*sizeof(data_type));
  for( unsigned i = 0; i < 8*sizeof(data_type); ++i ) {
    result[i] = static_cast<bool>((data.payload >> i) & 0x1);
  }
  return result;
}

HGCal64BitRandomCodec::data_type
HGCal64BitRandomCodec::
decodeImpl(const std::vector<bool>& data) const {
  data_type result;
  result.payload = 0;
  if( data.size() > 8*sizeof(data_type) ) {
    edm::LogWarning("HGCal64BitRandomCodec|TruncateInput")
          << "Input to be encoded was larger than data size: "
          << sizeof(data_type) << ". Truncating to fit!";
  }  
  for( unsigned i = 0; i < 8*sizeof(data_type); ++i ) {
    result.payload |= static_cast<uint64_t>(data[i]) << i;
  }
  return result;
}


