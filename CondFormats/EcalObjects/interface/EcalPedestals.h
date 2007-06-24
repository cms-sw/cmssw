#ifndef EcalPedestals_h
#define EcalPedestals_h

#include <map>
#include <boost/cstdint.hpp>

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
  std::map<uint32_t, Item> m_pedestals;

  void update() const;

  Item const & barrel(size_t hashid) const {
    return m_barrel[hashid];
  }
  Item const & endcap(size_t hashid) const {
    return m_endcap[hashid];
  }

private:
  mutable std::vector<Item> m_barrel;
  mutable std::vector<Item> m_endcap;

};

typedef std::map<uint32_t, EcalPedestals::Item>                 EcalPedestalsMap;
typedef std::map<uint32_t, EcalPedestals::Item>::const_iterator EcalPedestalsMapIterator;

#endif
