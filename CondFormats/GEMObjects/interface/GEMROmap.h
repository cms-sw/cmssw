#ifndef CondFormats_GEMObjects_GEMROmap_h
#define CondFormats_GEMObjects_GEMROmap_h
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include <map>

class GEMROmap{
 public:
  
  struct eCoord{
    uint16_t amcId;
    uint16_t gebId;
    uint16_t vfatId;
    bool operator < (const eCoord& r) const{
      if (amcId == r.amcId){
        if ( gebId == r.gebId){
          return vfatId < r.vfatId;
        }else{
          return gebId < r.gebId;
         }
      }else{
	return amcId < r.amcId;
      }
    }
  };
  
  struct dCoord{
    int vfatType;
    GEMDetId gemDetId;
    int iPhi;
    bool operator < (const dCoord& r) const{
      if (vfatType == r.vfatType){
        if (gemDetId == r.gemDetId){
	  return iPhi < r.iPhi;
        }else{
          return gemDetId < r.gemDetId;
        }
      }else{
	return vfatType < r.vfatType;
      }
    }    
  };

  struct channelNum{
    int vfatType;
    int chNum;
    bool operator < (const channelNum& c) const{
      if (vfatType == c.vfatType)
        return chNum < c.chNum;
      else
        return vfatType < c.vfatType;
    }
  };

  struct stripNum{
    int vfatType;
    int stNum;
    bool operator < (const stripNum& s) const{
      if (vfatType == s.vfatType) 
        return stNum < s.stNum;
      else
        return vfatType < s.vfatType;
    }
  };

  GEMROmap(){};
  
  bool isValidChipID(const eCoord& r) const {
    return roMapED_.find(r) != roMapED_.end();
  }
  const dCoord& hitPosition(const eCoord& r) const {return roMapED_.at(r);}
  const eCoord& hitPosition(const dCoord& r) const {return roMapDE_.at(r);}
  
  void add(eCoord e,dCoord d) {roMapED_[e]=d;}
  void add(dCoord d,eCoord e) {roMapDE_[d]=e;}

  const uint32_t& gebPosition(const GEMDetId& r) const {return roMapDet2Geb_.at(r);}
  const GEMDetId& gebPosition(const uint32_t& r) const {return roMapGeb2Det_.at(r);}
  
  void add(GEMDetId d,uint32_t e) {roMapDet2Geb_[d]=e;}
  void add(uint32_t d,GEMDetId e) {roMapGeb2Det_[d]=e;}
  
  std::map<GEMDetId,uint32_t> getRoMap(){return roMapDet2Geb_;}

  void addAMC(uint16_t d) {amcs_.push_back(d);}
  std::vector<uint16_t> getAMCs() const {return amcs_;}

  void addAMC2GEB(uint16_t d, uint16_t c) {amc2Gebs_[d].push_back(c);}
  std::vector<uint16_t> getAMC2GEBs(uint16_t d) const {return amc2Gebs_.at(d);}

  void add(channelNum c, stripNum s) {chStMap_[c]=s;} 
  void add(stripNum s, channelNum c) {stChMap_[s]=c;} 
 
  const channelNum& hitPosition(const stripNum& s) const {return stChMap_.at(s);}
  const stripNum& hitPosition(const channelNum& c) const {return chStMap_.at(c);}

 private:
  std::vector<uint16_t> amcs_;
  std::map<uint16_t,std::vector<uint16_t>> amc2Gebs_;
  
  std::map<eCoord,dCoord> roMapED_;
  std::map<dCoord,eCoord> roMapDE_;

  std::map<GEMDetId,uint32_t> roMapDet2Geb_;
  std::map<uint32_t,GEMDetId> roMapGeb2Det_;
   
  std::map<channelNum, stripNum> chStMap_;
  std::map<stripNum, channelNum> stChMap_;
  
};
#endif
