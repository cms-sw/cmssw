#ifndef __L1Trigger_L1THGCal_HGCal64BitRandomCodec_h__
#define __L1Trigger_L1THGCal_HGCal64BitRandomCodec_h__

#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCal64BitRandomCodecImpl.h"
#include <limits>


inline std::ostream& operator<<(std::ostream& o, 
                                const HGCal64BitRandomDataPayload& data) { 
  o << std::hex << data.payload << std::dec;
  return o;
}

class HGCal64BitRandomCodec : public HGCalTriggerFE::Codec<HGCal64BitRandomCodec,HGCal64BitRandomDataPayload> {
public:
  typedef HGCal64BitRandomDataPayload data_type;
  
  HGCal64BitRandomCodec(const edm::ParameterSet& conf) :
    Codec(conf),
    codecImpl_(conf) {
    data_.payload = std::numeric_limits<uint64_t>::max();
  }

  void setDataPayloadImpl(const HGCEEDigiCollection& ee,
                          const HGCHEDigiCollection& fh,
                          const HGCHEDigiCollection& bh );

  void setDataPayloadImpl(const l1t::HGCFETriggerDigi& digi);
  
  std::vector<bool> encodeImpl(const data_type&) const ;
  data_type         decodeImpl(const std::vector<bool>&, const uint32_t) const;  

private:
  HGCal64BitRandomCodecImpl codecImpl_;
};

#endif
