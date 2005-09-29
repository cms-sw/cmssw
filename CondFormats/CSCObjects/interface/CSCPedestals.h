#ifndef CSCPedestals_h
#define CSCPedestals_h

#include <map>

class CSCPedestals{
 public:
  CSCPedestals();
  ~CSCPedestals();
  
  struct Item{
    int m_ped0;
    int m_ped1;
    float m_rms0;
    float m_rms1;
  };
  std::map<int, Item> m_pedestals;
};

#endif
