#ifndef CSCDBGains_h
#define CSCDBGains_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <iosfwd>

class CSCDBGains{
 public:
  CSCDBGains();
  ~CSCDBGains();
  
  struct Item{
    short int gain_slope;
  };
  enum size{ArraySize=217728};

  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef Item GainContainer;
  GainContainer gains[ArraySize];
};

std::ostream & operator<<(std::ostream & os, const CSCDBGains & cscDbGains);

#endif

