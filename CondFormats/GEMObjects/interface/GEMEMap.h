#ifndef GEMEMap_H
#define GEMEMap_H
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
};

#endif // GEMEMap_H
