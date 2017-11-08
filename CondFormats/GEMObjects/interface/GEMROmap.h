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

  bool isValidChipID(const eCoord& r){
    return roMapED_.find(r) != roMapED_.end();
  }
  const dCoord& hitPosition(const eCoord& r){return roMapED_[r];}
  const eCoord& hitPosition(const dCoord& r){return roMapDE_[r];}
  
  void add(eCoord e,dCoord d) {roMapED_[e]=d;}
  void add(dCoord d,eCoord e) {roMapDE_[d]=e;}
  
 private:
  std::map<eCoord,dCoord> roMapED_;
  std::map<dCoord,eCoord> roMapDE_;
  
};
#endif
