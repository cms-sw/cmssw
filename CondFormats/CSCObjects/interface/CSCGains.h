#ifndef CSCGains_h
#define CSCGains_h

#include <map>

class CSCGains{
 public:
  CSCGains();
  ~CSCGains();
  
  struct Item{
    float slope;
    float intercept;
  };
  std::map<int, Item> gains;
};

#endif

