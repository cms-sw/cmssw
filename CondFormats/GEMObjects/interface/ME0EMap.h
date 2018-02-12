#ifndef CondFormats_GEMObjects_ME0EMap_h
#define CondFormats_GEMObjects_ME0EMap_h

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
  void convert(ME0ROmap & romap);
  void convertDummy(ME0ROmap & romap);

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
    std::vector<int> vfatType;
    std::vector<uint16_t> vfatId;
    std::vector<uint16_t> amcId;
    std::vector<uint16_t> gebId;
    std::vector<int> sec; 

    COND_SERIALIZABLE;
  };
  struct ME0ChStripMap {

    std::vector<int> vfatType;
    std::vector<int> vfatCh;
    std::vector<int> vfatStrip;
 
    COND_SERIALIZABLE;
  };

  struct ME0VFatMapInPos {
    int position;
    int VFATmapTypeId;

    COND_SERIALIZABLE;
  };

  std::vector<ME0EMapItem>     theEMapItem_;
  std::vector<ME0VFatMaptype>  theVFatMaptype_;
  std::vector<ME0VFatMapInPos> theVFatMapInPos_;
  std::vector<ME0ChStripMap>   theVfatChStripMap_;
  
 private:
  std::string theVersion;

  COND_SERIALIZABLE;
  
 public:
  // size of ID bits
  static const int chipIdBits_ = 12;     // ID size from VFat
  static const int chipIdMask_ = 0xfff;  // chipId mask for 12 bits
  static const int gebIdBits_  = 5;      // ID size from GEB
  static const int maxGEBs_    = 24;     // 24 gebs per amc
  static const int maxVFat_    = 6;      // vFat per eta partition in ME0, not known yet
  static const int maxChan_    = 128;    // channels per vFat
  static const int amcBX_      = 25;     // amc BX to get strip bx
};
#endif // ME0EMap_H
