#ifndef CSCChamberIndex_h
#define CSCChamberIndex_h

#include <vector>
#include <string>

class CSCChamberIndex{
 public:
  CSCChamberIndex();
  ~CSCChamberIndex();
  
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

  typedef std::vector< MapItem > CSCVector;
  CSCVector ch_index;
};

#endif
