#ifndef CSCcrosstalk_h
#define CSCcrosstalk_h

#include <vector>
#include <map>

class CSCcrosstalk{
 public:
  CSCcrosstalk();
  ~CSCcrosstalk();
  
  struct Item{
    float xtalk_slope_right;
    float xtalk_intercept_right;
    float xtalk_chi2_right;
    float xtalk_slope_left;
    float xtalk_intercept_left;
    float xtalk_chi2_left;
  };
  std::map< int,std::vector<Item> > crosstalk;
};

#endif

