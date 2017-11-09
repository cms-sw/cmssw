#ifndef ME0RawToDigi_ME0ROMAO_H
#define ME0RawToDigi_ME0ROMAO_H
#include <map>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>

class ME0ROmap{
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
    ME0DetId me0DetId;
    bool operator < (const dCoord& r) const{
      if ( me0DetId == r.me0DetId)
	return stripId < r.stripId;
      else
	return me0DetId < r.me0DetId;
    }    
  };

  ME0ROmap(){};

  bool isValidChipID(const eCoord& r){
    return roMapED_.find(r) != roMapED_.end();
  }
  const dCoord& hitPosition(const eCoord& r){return roMapED_[r];}
  const eCoord& hitPosition(const dCoord& r){return roMapDE_[r];}
  
  void add(eCoord e,dCoord d) {roMapED_[e]=d;}
  void add(dCoord d,eCoord e) {roMapDE_[d]=e;}

  const int& gebPosition(const ME0DetId& r){return roMapDet2Geb_[r];}
  const ME0DetId& gebPosition(const int& r){return roMapGeb2Det_[r];}
  
  void add(ME0DetId d,int e) {roMapDet2Geb_[d]=e;}
  void add(int d,ME0DetId e) {roMapGeb2Det_[d]=e;}
  
 private:
  std::map<eCoord,dCoord> roMapED_;
  std::map<dCoord,eCoord> roMapDE_;

  std::map<ME0DetId,int> roMapDet2Geb_;
  std::map<int,ME0DetId> roMapGeb2Det_;
  
};
#endif
