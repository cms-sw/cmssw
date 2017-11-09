#ifndef GEMRawToDigi_ME0ROMAO_H
#define GEMRawToDigi_ME0ROMAO_H
#include <map>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>

class ME0ROmap{
 public:
  
  struct eCoord{
    uint32_t vfatId;
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

  const uint32_t& gebPosition(const ME0DetId& r){return roMapDet2Geb_[r];}
  const ME0DetId& gebPosition(const uint32_t& r){return roMapGeb2Det_[r];}
  
  void add(ME0DetId d,uint32_t e) {roMapDet2Geb_[d]=e;}
  void add(uint32_t d,ME0DetId e) {roMapGeb2Det_[d]=e;}
  
  std::map<ME0DetId,uint32_t> getRoMap(){return roMapDet2Geb_;}

  void addAMC(uint16_t d) {amcs_.push_back(d);}
  std::vector<uint16_t> getAMCs(){return amcs_;}

  void addAMC2GEB(uint16_t d, uint16_t c) {amc2Gebs_[d].push_back(c);}
  std::vector<uint16_t> getAMC2GEBs(uint16_t d){return amc2Gebs_[d];}
  
 private:
  std::vector<uint16_t> amcs_;
  std::map<uint16_t,std::vector<uint16_t>> amc2Gebs_;
  
  std::map<eCoord,dCoord> roMapED_;
  std::map<dCoord,eCoord> roMapDE_;

  std::map<ME0DetId,uint32_t> roMapDet2Geb_;
  std::map<uint32_t,ME0DetId> roMapGeb2Det_;
  
};
#endif
