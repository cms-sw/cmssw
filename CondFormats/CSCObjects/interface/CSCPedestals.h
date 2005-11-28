#ifndef CSCPedestals_h
#define CSCPedestals_h

#include <vector>

class CSCPedestals{
 public:
  CSCPedestals();
  ~CSCPedestals();
  
  struct Item{
    float ped;
    float rms;
  };
  std::vector<Item> pedestals;
};

#endif
