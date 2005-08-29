#ifndef EcalPedestals_h
#define EcalPedestals_h

#include <map>

class EcalPedestals {
 public:
  EcalPedestals();
  ~EcalPedestals();
  struct Item {
    float m_mean;
    float m_variance;
  };
  std::map<int, Item> m_pedestals;
};

#endif
