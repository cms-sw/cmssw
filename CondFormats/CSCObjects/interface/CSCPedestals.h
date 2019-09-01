#ifndef CSCPedestals_h
#define CSCPedestals_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <map>

class CSCPedestals {
public:
  CSCPedestals();
  ~CSCPedestals();

  struct Item {
    float ped;
    float rms;

    COND_SERIALIZABLE;
  };

  const Item& item(const CSCDetId& cscId, int strip) const;

  typedef std::map<int, std::vector<Item> > PedestalMap;
  PedestalMap pedestals;

  COND_SERIALIZABLE;
};

#endif
