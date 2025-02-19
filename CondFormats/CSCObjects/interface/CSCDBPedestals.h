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
  int factor_ped;
  int factor_rms;

  enum factors{FPED=10, FRMS=1000};

  // accessor to appropriate element
  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef std::vector<Item> PedestalContainer;

  PedestalContainer pedestals;
};

#endif
