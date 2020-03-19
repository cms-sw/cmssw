#ifndef CondFormats_GEMObjects_GEMROmap_h
#define CondFormats_GEMObjects_GEMROmap_h
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include <map>

class GEMROmap {
public:
  struct eCoord {
    uint16_t amcId;
    uint16_t gebId;
    uint16_t vfatId;
    bool operator<(const eCoord& r) const {
      if (amcId == r.amcId) {
        if (gebId == r.gebId) {
          return vfatId < r.vfatId;
        } else {
          return gebId < r.gebId;
        }
      } else {
        return amcId < r.amcId;
      }
    }
  };

  struct dCoord {
    int vfatType;
    GEMDetId gemDetId;
    int iPhi;
    bool operator<(const dCoord& r) const {
      if (vfatType == r.vfatType) {
        if (gemDetId == r.gemDetId) {
          return iPhi < r.iPhi;
        } else {
          return gemDetId < r.gemDetId;
        }
      } else {
        return vfatType < r.vfatType;
      }
    }
  };

  struct channelNum {
    int vfatType;
    int chNum;
    bool operator<(const channelNum& c) const {
      if (vfatType == c.vfatType)
        return chNum < c.chNum;
      else
        return vfatType < c.vfatType;
    }
  };

  struct stripNum {
    int vfatType;
    int stNum;
    bool operator<(const stripNum& s) const {
      if (vfatType == s.vfatType)
        return stNum < s.stNum;
      else
        return vfatType < s.vfatType;
    }
  };

  GEMROmap(){};

  bool isValidChipID(const eCoord& r) const { return roMapED_.find(r) != roMapED_.end(); }
  const dCoord& hitPosition(const eCoord& r) const { return roMapED_.at(r); }
  const eCoord& hitPosition(const dCoord& r) const { return roMapDE_.at(r); }

  void add(eCoord e, dCoord d) { roMapED_[e] = d; }
  void add(dCoord d, eCoord e) { roMapDE_[d] = e; }

  const std::map<eCoord, dCoord>* getRoMap() const { return &roMapED_; }

  void add(channelNum c, stripNum s) { chStMap_[c] = s; }
  void add(stripNum s, channelNum c) { stChMap_[s] = c; }

  const channelNum& hitPosition(const stripNum& s) const { return stChMap_.at(s); }
  const stripNum& hitPosition(const channelNum& c) const { return chStMap_.at(c); }

private:
  std::map<eCoord, dCoord> roMapED_;
  std::map<dCoord, eCoord> roMapDE_;

  std::map<channelNum, stripNum> chStMap_;
  std::map<stripNum, channelNum> stChMap_;
};
#endif
