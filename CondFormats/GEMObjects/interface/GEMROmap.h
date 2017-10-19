#ifndef GEMRawToDigi_GEMROMAO_H
#define GEMRawToDigi_GEMROMAO_H
#include <map>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>

class GEMROmap{
 public:
  
  struct eCoord{
    uint16_t vfatId;
    int channelId;
    bool operator < (const eCoord& r) const{
      if ( vfatId == r.vfatId)
	return channelId < r.channelId;
      else
	return vfatId < r.vfatId;
    }
  };
  
  struct dCoord{
    int stripId;
    GEMDetId gemDetId;
    bool operator < (const dCoord& r) const{
      if ( gemDetId == r.gemDetId)
	return stripId < r.stripId;
      else
	return gemDetId < r.gemDetId;
    }    
  };

  GEMROmap(){};

  bool isValidChipID(const eCoord& ec){
    return roMap.find(ec) != roMap.end();
  }
  const dCoord& hitPosition(const eCoord& ec){return roMap[ec];}
  
  void add(eCoord e,dCoord d) {roMap[e]=d;}
  
 private:
  std::map<eCoord,dCoord> roMap;
  
};
#endif
