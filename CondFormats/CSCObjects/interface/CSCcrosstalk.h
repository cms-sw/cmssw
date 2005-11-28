#ifndef CSCcrosstalk_h
#define CSCcrosstalk_h

#include <vector>

class CSCcrosstalk{
 public:
  CSCcrosstalk();
  ~CSCcrosstalk();
  
  struct Item{
    float slope;
    float intercept;
  };
  std::vector<Item> crosstalk;
};

#endif

