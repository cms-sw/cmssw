#ifndef GEMMapItem_h
#define GEMMapItem_h
// based on CSCMapItem
#include "CondFormats/Serialization/interface/Serializable.h"
#include <string>

class GEMMapItem{
 public:
  GEMMapItem();
  ~GEMMapItem();

  struct MapItem{
    std::string chamberLabel;
    int chamberId;
    int endcap;
    int station;
    int ring;
    int chamber;
    int gemIndex;
    int layerIndex;
    int stripIndex;
    std::string crateLabel;
    int crateid;
    int sector;
    int trig_sector;
    int dmb;
    int gemid;
    int ddu;
    int ddu_input;
    int slink;
    int fed_crate;
    int ddu_slot;
    std::string dcc_fifo;
    int fiber_crate;
    int fiber_pos;
    std::string fiber_socket;
  
    COND_SERIALIZABLE;
  };

  COND_SERIALIZABLE;
};

#endif
