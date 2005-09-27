#ifndef EcalPedestals_h
#define EcalPedestals_h

#include <map>

class EcalPedestals {
 public:
  EcalPedestals();
  ~EcalPedestals();
  struct Item {
    float mean_x12;
    float rms_x12;
    float mean_x6;
    float rms_x6;
    float mean_x1;
    float rms_x1;
  };
  std::map<int, Item> m_pedestals;
};

typedef std::map<int, EcalPedestals::Item>                 EcalPedestalsMap;
typedef std::map<int, EcalPedestals::Item>::const_iterator EcalPedestalsMapIterator;

#endif
