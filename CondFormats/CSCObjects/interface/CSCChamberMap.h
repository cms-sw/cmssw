#ifndef CSCChamberMap_h
#define CSCChamberMap_h

#include <vector>
#include <map>
#include <string>

class CSCChamberMap{
 public:
  CSCChamberMap();
  ~CSCChamberMap();
  
  struct MapItem{
    std::string chamberLabel;
    int chamberId;
    int endcap;
    int station;
    int ring;
    int chamber;
    int cscIndex;
    int layerIndex;
    int stripIndex;
    int anodeIndex;
    int strips;
    int anodes;
    std::string crateLabel;
    int crateid;
    int sector;
    int trig_sector;
    int dmb;
    int cscid;
    int ddu;
    int ddu_input;
    int slink;
  };

  typedef std::map< int,MapItem > ChamberMap;
  ChamberMap ch_map;
};

#endif

