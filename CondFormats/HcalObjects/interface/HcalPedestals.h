#ifndef HcalPedestals_h
#define HcalPedestals_h

#include <map>

class HcalPedestals {
 public:
  HcalPedestals();
  ~HcalPedestals();
  struct Item {
    float m_mean1;
    float m_variance1;
    float m_mean2;
    float m_variance2;
    float m_mean3;
    float m_variance3;
    float m_mean4;
    float m_variance4;
  };
  std::map<int, Item> m_pedestals;
};

#endif
