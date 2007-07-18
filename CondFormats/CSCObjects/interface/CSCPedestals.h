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

  const Item & item(int cscId, int strip) const;

  typedef std::map< int,std::vector<Item> > PedestalMap;
  PedestalMap pedestals;
};

#endif
