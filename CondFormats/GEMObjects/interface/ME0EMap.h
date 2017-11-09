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
};

#endif // ME0EMap_H
