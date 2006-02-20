#ifndef CSCGains_h
#define CSCGains_h

#include <vector>

class CSCGains{
 public:
  CSCGains();
  ~CSCGains();
  
  struct Item{
    float gain_slope;
    float gain_intercept;
  };
  std::vector<Item> gains;
};

#endif

