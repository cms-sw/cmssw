#ifndef CSCDBGains_h
#define CSCDBGains_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <iosfwd>
#include <vector>

class CSCDBGains{
 public:


  CSCDBGains();
  ~CSCDBGains();
  
  struct Item{
    short int gain_slope;
  };

  // accessor to appropriate element
  const Item & item(const CSCDetId & cscId, int strip) const;
  
  typedef std::vector<Item> GainContainer;

  GainContainer gains;
};

std::ostream & operator<<(std::ostream & os, const CSCDBGains & cscDbGains);

#endif

