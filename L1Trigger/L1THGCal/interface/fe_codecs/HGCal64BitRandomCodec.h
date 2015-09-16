#ifndef __L1Trigger_L1THGCal_HGCal64BitRandomCodec_h__
#define __L1Trigger_L1THGCal_HGCal64BitRandomCodec_h__

#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include <limits>

#include "TRandom3.h"

struct HGCal64BitRandomDataPayload { 
  uint64_t payload;
  void reset() { memset(&payload,0,sizeof(uint64_t)); }
};

inline std::ostream& operator<<(std::ostream& o, 
                                const HGCal64BitRandomDataPayload& data) { 
  o << std::hex << data.payload << std::dec;
  return o;
}

class HGCal64BitRandomCodec : public HGCalTriggerFE::Codec<HGCal64BitRandomCodec,HGCal64BitRandomDataPayload> {
public:
  typedef HGCal64BitRandomDataPayload data_type;
  
  HGCal64BitRandomCodec(const edm::ParameterSet& conf) :
    Codec(conf) {
    data_.payload = std::numeric_limits<uint64_t>::max();
    rand.SetSeed(0);
  }

  void setDataPayloadImpl(const Module& mod, 
                          const HGCalTriggerGeometryBase& geom,
                          const HGCEEDigiCollection& ee,
                          const HGCHEDigiCollection& fh,
                          const HGCHEDigiCollection& bh );
  
  std::vector<bool> encodeImpl(const data_type&) const ;
  data_type         decodeImpl(const std::vector<bool>&) const;  

private:
  TRandom3 rand;
};

#endif
