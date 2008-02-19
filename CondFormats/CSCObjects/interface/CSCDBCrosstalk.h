#ifndef CSCObjects_CSCDBCrosstalk_h
#define CSCObjects_CSCDBCrosstalk_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCDBCrosstalk
{
 public:
  CSCDBCrosstalk() {}
  ~CSCDBCrosstalk() {}
  
  struct Item{
    short int xtalk_slope_right;
    short int xtalk_intercept_right;
    short int xtalk_slope_left;
    short int xtalk_intercept_left;
  };
  enum size{ArraySize=217728};

  const Item & item(const CSCDetId & cscId, int strip) const;

  typedef Item CrosstalkContainer;
  CrosstalkContainer crosstalk[ArraySize];
};

#endif

