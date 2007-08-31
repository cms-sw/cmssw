#ifndef CSCDBGains_h
#define CSCDBGains_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <map>

class CSCDBGains{
 public:
  CSCDBGains();
  ~CSCDBGains();
  
  struct Item{
    float gain_slope;
    float gain_intercept;
    float gain_chi2;
  };

  const Item & item(const CSCDetId & cscId, int strip) const;

  //typedef std::map< int,std::vector<Item> > GainsMap;
  typedef std::vector<Item> GainsContainer;
  GainsContainer gains;
};

#endif

