#ifndef GEMRawToDigi_GEMROMAO_H
#define GEMRawToDigi_GEMROMAO_H
#include <map>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>

class GEMROmap{
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

  const uint32_t& gebPosition(const GEMDetId& r){return roMapDet2Geb_[r];}
  const GEMDetId& gebPosition(const uint32_t& r){return roMapGeb2Det_[r];}
  
  void add(GEMDetId d,uint32_t e) {roMapDet2Geb_[d]=e;}
  void add(uint32_t d,GEMDetId e) {roMapGeb2Det_[d]=e;}
  
  std::map<GEMDetId,uint32_t> getRoMap(){return roMapDet2Geb_;}

  void addAMC(uint16_t d) {amcs_.push_back(d);}
  std::vector<uint16_t> getAMCs(){return amcs_;}

  void addAMC2GEB(uint16_t d, uint16_t c) {amc2Gebs_[d].push_back(c);}
  std::vector<uint16_t> getAMC2GEBs(uint16_t d){return amc2Gebs_[d];}
  
 private:
  std::vector<uint16_t> amcs_;
  std::map<uint16_t,std::vector<uint16_t>> amc2Gebs_;
  
  std::map<eCoord,dCoord> roMapED_;
  std::map<dCoord,eCoord> roMapDE_;

  std::map<GEMDetId,uint32_t> roMapDet2Geb_;
  std::map<uint32_t,GEMDetId> roMapGeb2Det_;
  
};
#endif
