#ifndef SiStripPedestals_h
#define SiStripPedestals_h

#include<vector>
#include <map>

class SiStripPedestals {
 public:
  SiStripPedestals();
  ~SiStripPedestals();

  struct Item {
    short int ped;
    short int noise;
    //int lowTh;
    //int highTh;
    bool disabled;
  };

  typedef std::vector<Item>                 SiStripPedestalsVector;
  typedef std::vector<Item>::const_iterator SiStripPedestalsVectorIterator;
  
  typedef std::map<int, SiStripPedestalsVector>                 SiStripPedestalsMap;
  typedef std::map<int, SiStripPedestalsVector>::const_iterator SiStripPedestalsMapIterator;
  std::map<int, SiStripPedestalsVector> m_pedestals;
};
#endif
