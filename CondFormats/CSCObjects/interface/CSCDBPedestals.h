#ifndef CSCDBPedestals_h
#define CSCDBPedestals_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCDBPedestals{
 public:
  CSCDBPedestals();
  ~CSCDBPedestals();
  
  struct Item{
    short int ped;
    short int rms;
  };
  enum size{ArraySize=217728};

  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef Item PedestalContainer;
  PedestalContainer pedestals[ArraySize];
};

#endif
