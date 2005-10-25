#ifndef CSCPedestals_h
#define CSCPedestals_h

#include <map>

class CSCPedestals{
 public:
  CSCPedestals();
  ~CSCPedestals();
  
  struct Item{
    float ped;
    float rms;
  };
  std::map<int, Item> pedestals;
};

#endif
