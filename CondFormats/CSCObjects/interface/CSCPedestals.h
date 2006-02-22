#ifndef CSCPedestals_h
#define CSCPedestals_h

#include <vector>
#include <map>

class CSCPedestals{
 public:
  CSCPedestals();
  ~CSCPedestals();
  
  struct Item{
    float ped;
    float rms;
  };
  std::map< int,std::vector<Item> > pedestals;
};

#endif
