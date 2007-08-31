#ifndef CSCDBPedestals_h
#define CSCDBPedestals_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <map>

class CSCDBPedestals{
 public:
  CSCDBPedestals();
  ~CSCDBPedestals();
  
  struct Item{
    float ped;
    float rms;
  };

  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef std::vector<Item> PedestalContainer;
  PedestalContainer pedestals;
};

#endif
