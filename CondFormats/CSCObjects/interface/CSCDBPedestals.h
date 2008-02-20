#ifndef CSCDBPedestals_h
#define CSCDBPedestals_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>

class CSCDBPedestals{
 public:
  CSCDBPedestals();
  ~CSCDBPedestals();
  
  struct Item{
    short int ped;
    short int rms;
  };

  // accessor to appropriate element
  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef std::vector<Item> PedestalContainer;

  PedestalContainer pedestals;
};

#endif
