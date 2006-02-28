#ifndef CSCGains_h
#define CSCGains_h

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
  std::map< int,std::vector<Item> > gains;
};

#endif

