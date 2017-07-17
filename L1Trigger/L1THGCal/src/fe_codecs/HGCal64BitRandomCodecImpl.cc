#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCal64BitRandomCodecImpl.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <limits>
#include "TRandom3.h"


void HGCal64BitRandomCodecImpl::
setDataPayload(data_type& data) {
  data.payload = 0;
  for( unsigned i = 0; i < 8*sizeof(data_type); ++i ) {
    data.payload |= static_cast<uint64_t>(rand_.Rndm() > 0.5) << i;
  }
}

std::vector<bool>
HGCal64BitRandomCodecImpl::
encode(const HGCal64BitRandomCodecImpl::data_type& data) const {
  std::vector<bool> result;
  result.resize(8*sizeof(data_type));
  for( unsigned i = 0; i < 8*sizeof(data_type); ++i ) {
    result[i] = static_cast<bool>((data.payload >> i) & 0x1);
  }
  return result;
}

HGCal64BitRandomCodecImpl::data_type
HGCal64BitRandomCodecImpl::
decode(const std::vector<bool>& data) const {
  data_type result;
  result.payload = 0;
  if( data.size() > 8*sizeof(data_type) ) {
    edm::LogWarning("HGCal64BitRandomCodecImpl|TruncateInput")
          << "Input to be encoded was larger than data size: "
          << sizeof(data_type) << ". Truncating to fit!";
  }  
  for( unsigned i = 0; i < 8*sizeof(data_type); ++i ) {
    result.payload |= static_cast<uint64_t>(data[i]) << i;
  }
  return result;
}


