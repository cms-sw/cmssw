#ifndef CSCGains_h
#define CSCGains_h

#include <vector>

class CSCGains{
 public:
  CSCGains();
  ~CSCGains();
  
  struct Item{
    float slope;
    float intercept;
  };
  std::vector<Item> gains;
};

#endif

