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
  int ped_factor;
  int rms_factor;

  enum factors{PedFactor=10, RmsFactor=1000};

  // accessor to appropriate element
  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef std::vector<Item> PedestalContainer;

  PedestalContainer pedestals;
};

#endif
