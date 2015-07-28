#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include <limits>

#include "TRandom3.h"

using namespace HGCalTriggerFE;

struct HGCal64BitRandomDataPayload {
 
  unsigned int payload;
};

std::ostream& operator<<(std::ostream& o, 
                         const HGCal64BitRandomDataPayload& data) { 
  o << std::hex << data.payload << std::dec;
  return o;
}

class HGCal64BitRandomCodec : public Codec<HGCal64BitRandomCodec,HGCal64BitRandomDataPayload> {
public:
  typedef HGCal64BitRandomDataPayload data_type;
  
  HGCal64BitRandomCodec(const edm::ParameterSet& conf) :
    Codec(conf) {
    data_.payload = std::numeric_limits<unsigned int>::max();
    rand.SetSeed(0);
  }

  void setDataPayloadImpl(const Module& mod, 
                          const HGCEEDigiCollection& ee,
                          const HGCHEDigiCollection& fh,
                          const HGCHEDigiCollection& bh );
  
  std::vector<bool> encodeImpl(const data_type&) const ;
  data_type         decodeImpl(const std::vector<bool>&) const;  

private:
  TRandom3 rand;
};

void HGCal64BitRandomCodec::
setDataPayloadImpl(const Module& , 
                   const HGCEEDigiCollection&,
                   const HGCHEDigiCollection&,
                   const HGCHEDigiCollection& ) {
  data_.payload = 0;
  for( unsigned i = 0; i < sizeof(data_type); ++i ) {
    data_.payload |= static_cast<unsigned int>(rand.Rndm() > 0.5) << i;
  }
}

std::vector<bool>
HGCal64BitRandomCodec::
encodeImpl(const HGCal64BitRandomCodec::data_type& data) const {
  std::vector<bool> result;
  result.resize(sizeof(data_type));
  for( unsigned i = 0; i < sizeof(data_type); ++i ) {
    result[i] = static_cast<bool>((data.payload >> i) & 0x1);
  }
  return result;
}

HGCal64BitRandomCodec::data_type
HGCal64BitRandomCodec::
decodeImpl(const std::vector<bool>& data) const {
  data_type result;
  result.payload = 0;
  if( data.size() > sizeof(data_type) ) {
    edm::LogWarning("HGCal64BitRandomCodec|TruncateInput")
          << "Input to be encoded was larger than data size: "
          << sizeof(data_type) << ". Truncating to fit!";
  }  
  for( unsigned i = 0; i < sizeof(data_type); ++i ) {
    result.payload |= static_cast<unsigned int>(data[i]) << i;
  }
  return result;
}

DEFINE_EDM_PLUGIN(HGCalTriggerFECodecFactory, 
                  HGCal64BitRandomCodec,
                  "HGCal64BitRandomCodec");
