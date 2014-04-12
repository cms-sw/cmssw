#ifndef CSCGains_h
#define CSCGains_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <map>

class CSCGains{
 public:
  CSCGains();
  ~CSCGains();
  
  struct Item{
    float gain_slope;
    float gain_intercept;
    float gain_chi2;
  };

  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef std::map< int,std::vector<Item> > GainsMap;
  GainsMap gains;
};

#endif

