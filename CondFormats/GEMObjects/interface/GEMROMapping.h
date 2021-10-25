#ifndef CondFormats_GEMObjects_GEMROMapping_h
#define CondFormats_GEMObjects_GEMROMapping_h
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include <map>
#include <vector>
#include <algorithm>

class GEMROMapping {
  // EC electronics corrdinate
  // DC GEMDetId corrdinate
  // geb = GEM electronics board == OptoHybrid
public:
  struct sectorEC {
    unsigned int fedId;
    uint8_t amcNum;
    bool operator==(const sectorEC& r) const {
      if (fedId == r.fedId) {
        return amcNum == r.amcNum;
      } else {
        return false;
      }
    }
  };

  struct chamEC {
    unsigned int fedId;
    uint8_t amcNum;
    uint8_t gebId;
    bool operator<(const chamEC& r) const {
      if (fedId == r.fedId) {
        if (amcNum == r.amcNum) {
          return gebId < r.gebId;
        } else {
          return amcNum < r.amcNum;
        }
      } else {
        return fedId < r.fedId;
      }
    }
  };

  struct chamDC {
    GEMDetId detId;
    int vfatVer;
    bool operator<(const chamDC& r) const { return detId < r.detId; }
  };

  struct vfatEC {
    uint16_t vfatAdd;
    GEMDetId detId;
    bool operator<(const vfatEC& r) const {
      if (vfatAdd == r.vfatAdd) {
        return detId < r.detId;
      } else {
        return vfatAdd < r.vfatAdd;
      }
    }
  };

  struct vfatDC {
    int vfatType;
    GEMDetId detId;
    int localPhi;
    bool operator<(const vfatDC& r) const {
      if (vfatType == r.vfatType) {
        if (detId == r.detId) {
          return localPhi < r.localPhi;
        } else {
          return detId < r.detId;
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

  GEMROMapping(){};

  bool isValidChipID(const vfatEC& r) const { return vfatMap_.find(r) != vfatMap_.end(); }
  bool isValidChamber(const chamEC& r) const { return chamberMap_.find(r) != chamberMap_.end(); }

  bool isValidAMC(const sectorEC& r) const { return std::find(amcVec_.begin(), amcVec_.end(), r) != amcVec_.end(); }

  void add(sectorEC e) { amcVec_.push_back(e); }

  const chamDC& chamberPos(const chamEC& r) const { return chamberMap_.at(r); }
  void add(chamEC e, chamDC d) { chamberMap_[e] = d; }

  const std::vector<vfatEC> getVfats(const GEMDetId& r) const { return chamVfats_.at(r); }
  void add(GEMDetId e, vfatEC d) { chamVfats_[e].push_back(d); }

  const vfatDC& vfatPos(const vfatEC& r) const { return vfatMap_.at(r); }
  void add(vfatEC e, vfatDC d) { vfatMap_[e] = d; }

  const channelNum& hitPos(const stripNum& s) const { return stChMap_.at(s); }
  const stripNum& hitPos(const channelNum& c) const { return chStMap_.at(c); }

  void add(channelNum c, stripNum s) { chStMap_[c] = s; }
  void add(stripNum s, channelNum c) { stChMap_[s] = c; }

private:
  std::vector<sectorEC> amcVec_;

  // electronics map to GEMDetId chamber
  std::map<chamEC, chamDC> chamberMap_;

  std::map<GEMDetId, std::vector<vfatEC>> chamVfats_;

  std::map<vfatEC, vfatDC> vfatMap_;

  std::map<channelNum, stripNum> chStMap_;
  std::map<stripNum, channelNum> stChMap_;
};
#endif
