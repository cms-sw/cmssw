#ifndef EventFilter_L1TRawToDigi_Omtf_LinkMappingCsc_H
#define EventFilter_L1TRawToDigi_Omtf_LinkMappingCsc_H

#include<map>
#include<cstdint>

#include "EventFilter/L1TRawToDigi/interface/OmtfEleIndex.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"


namespace omtf {
  typedef std::map<EleIndex, CSCDetId> MapEleIndex2CscDet;
  typedef std::map<uint32_t, std::pair<EleIndex,EleIndex> > MapCscDet2EleIndex; 

  MapEleIndex2CscDet mapEleIndex2CscDet();
  MapCscDet2EleIndex mapCscDet2EleIndex();

}
#endif
