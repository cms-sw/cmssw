#ifndef CSCGains_h
#define CSCGains_h

#include <map>

class CSCGains{
 public:
  CSCGains();
  ~CSCGains();
  
  struct Item{
    float m_slope;
    float m_intercept;
  };
  std::map<int, Item> m_gains;
};

#endif

