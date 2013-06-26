#ifndef CSCPedestals_h
#define CSCPedestals_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <map>

class CSCPedestals{
 public:
  CSCPedestals();
  ~CSCPedestals();
  
  struct Item{
    float ped;
    float rms;
  };

  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef std::map< int,std::vector<Item> > PedestalMap;
  PedestalMap pedestals;
};

#endif
