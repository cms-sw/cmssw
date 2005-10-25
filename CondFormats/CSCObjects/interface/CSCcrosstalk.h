#ifndef CSCcrosstalk_h
#define CSCcrosstalk_h

#include <map>

class CSCcrosstalk{
 public:
  CSCcrosstalk();
  ~CSCcrosstalk();
  
  struct Item{
    float slope;
    float intercept;
  };
  std::map<int, Item> crosstalk;
};

#endif

