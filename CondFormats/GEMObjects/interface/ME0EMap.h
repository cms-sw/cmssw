#ifndef ME0EMap_H
#define ME0EMap_H
#include "CondFormats/Serialization/interface/Serializable.h"
#include <string>
#include <vector>

class ME0ROmap;

class ME0EMap {
 public:
  ME0EMap();
  explicit ME0EMap(const std::string & version);

  virtual ~ME0EMap();

  const std::string & version() const;
  ME0ROmap* convert() const;
  ME0ROmap* convertDummy() const;

  struct ME0EMapItem {
    int ChamberID;
    std::vector<int> VFatIDs;
    std::vector<int> positions;

    COND_SERIALIZABLE;
  };  
  struct ME0VFatMaptype {
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
  struct ME0VFatMapInPos {
    int position;
    int VFATmapTypeId;

    COND_SERIALIZABLE;
  };

  std::vector<ME0EMapItem>     theEMapItem;
  std::vector<ME0VFatMaptype>  theVFatMaptype;
  std::vector<ME0VFatMapInPos> theVFatMapInPos;
  
 private:
  std::string theVersion;
  
  COND_SERIALIZABLE;

  // size of ID bits
  static const int chipIdBits_ = 12;     // ID size from VFat
  static const int chipIdMask_ = 0xfff;  // chipId mask for 12 bits
  static const int gebIdBits_  = 5;      // ID size from GEB
  static const int maxGEBs_    = 24;     // 24 gebs per amc
  static const int maxVFat_    = 12;     // vFat per eta partition, not known yet for ME0
  static const int maxChan_    = 128;    // channels per vFat
};
#endif // ME0EMap_H
