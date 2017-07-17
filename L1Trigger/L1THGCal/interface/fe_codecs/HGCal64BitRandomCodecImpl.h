#ifndef __L1Trigger_L1THGCal_HGCal64BitRandomCodecImpl_h__
#define __L1Trigger_L1THGCal_HGCal64BitRandomCodecImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TRandom3.h"

struct HGCal64BitRandomDataPayload { 
  uint64_t payload;
  void reset() { memset(&payload,0,sizeof(uint64_t)); }
};


class HGCal64BitRandomCodecImpl{
public:
  typedef HGCal64BitRandomDataPayload data_type;
  
  HGCal64BitRandomCodecImpl(const edm::ParameterSet& conf){
    rand_.SetSeed(0);
  }
  
  void setDataPayload(data_type&);
  std::vector<bool> encode(const data_type&) const ;
  data_type         decode(const std::vector<bool>&) const;  

private:
  TRandom3 rand_;

};

#endif
