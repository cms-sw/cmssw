#ifndef GEMEMap_h
#define GEMEMap_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <string>
#include <vector>

class GEMROmap;

class GEMEMap {
 public:
  GEMEMap();
  explicit GEMEMap(const std::string & version);

  virtual ~GEMEMap();

  const std::string & version() const;
  GEMROmap* convert() const;
  GEMROmap* convertDummy() const;

  struct GEMEMapItem {
    int ChamberID;
    std::vector<int> VFatIDs;
    std::vector<int> positions;

    COND_SERIALIZABLE;
  };  
  struct GEMVFatMaptype {
    int VFATmapTypeId;
    std::vector<int> vfat_position;
    std::vector<int> z_direction;
    std::vector<int> iEta;
    std::vector<int> iPhi;
    std::vector<int> depth;
    std::vector<int> strip_number;
    std::vector<int> vfat_chnnel_number;
    std::vector<uint16_t> vfatId;
    std::vector<int> sec; 

    COND_SERIALIZABLE;
  };
  struct GEMVFatMapInPos {
    int position;
    int VFATmapTypeId;

    COND_SERIALIZABLE;
  };

  std::vector<GEMEMapItem>     theEMapItem;
  std::vector<GEMVFatMaptype>  theVFatMaptype;
  std::vector<GEMVFatMapInPos> theVFatMapInPos;
  
 private:
  std::string theVersion;

  COND_SERIALIZABLE;

  // size of ID bits
  static const int chipIdBits_ = 12;     // ID size from VFat
  static const int chipIdMask_ = 0xfff;  // chipId mask for 12 bits
  static const int gebIdBits_  = 5;      // ID size from GEB
  static const int maxGEBs_    = 24;     // 24 gebs per amc
  static const int maxVFatGE11_= 3;     // vFat per eta partition in GE11
  static const int maxVFatGE21_= 6;     // vFat per eta partition in GE21
  static const int maxChan_    = 128;    // channels per vFat
};
#endif // GEMEMap_H
