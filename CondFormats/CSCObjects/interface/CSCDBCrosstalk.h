#ifndef CSCObjects_CSCDBCrosstalk_h
#define CSCObjects_CSCDBCrosstalk_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <map>

class CSCDBCrosstalk
{
 public:
  CSCDBCrosstalk() {}
  ~CSCDBCrosstalk() {}
  
  struct Item{
    float xtalk_slope_right;
    float xtalk_intercept_right;
    float xtalk_chi2_right;
    float xtalk_slope_left;
    float xtalk_intercept_left;
    float xtalk_chi2_left;
  };

  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef std::vector<Item> CrosstalkContainer;
  CrosstalkContainer crosstalk;
};

#endif

