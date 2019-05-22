#ifndef CSCObjects_CSCcrosstalk_h
#define CSCObjects_CSCcrosstalk_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <map>

class CSCcrosstalk {
public:
  CSCcrosstalk() {}
  ~CSCcrosstalk() {}

  struct Item {
    float xtalk_slope_right;
    float xtalk_intercept_right;
    float xtalk_chi2_right;
    float xtalk_slope_left;
    float xtalk_intercept_left;
    float xtalk_chi2_left;

    COND_SERIALIZABLE;
  };

  const Item& item(const CSCDetId& cscId, int strip) const;

  typedef std::map<int, std::vector<Item> > CrosstalkMap;
  CrosstalkMap crosstalk;

  COND_SERIALIZABLE;
};

#endif
