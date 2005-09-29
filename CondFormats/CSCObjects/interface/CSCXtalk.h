#ifndef CSCXtalk_h
#define CSCXtalk_h

#include <map>

class CSCXtalk{
 public:
  CSCXtalk();
  ~CSCXtalk();
  
  struct Item{
    float m_slope;
    float m_intercept;
    float m_xtalk_val;
  };
  std::map<int, Item> m_xtalk;
};

#endif

