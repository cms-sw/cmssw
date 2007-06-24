#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include <algorithm>

EcalPedestals::EcalPedestals(){}
EcalPedestals::~EcalPedestals(){}

namespace {
  template<typename DetId, typename Item>
  Inserter {
  public:
    Inserter(std::vector<Item> & iv) : v(iv){}

    void operator(){ std::pair<uint32_t, const Item> & p) {
      DetId id(p.first);
      if (id.null() || id.det()!=Ecal || id.subdetId()!=EcalBarrel ) 
	return;
      v.at(id.dhashedIndex()) = p.second;
    }

    std::vector<Item> & v;
  };
}

void EcalPedestals::update() const {
  if (m_barrel.empty()) {
    // FIXME
    m_barrel.resize(m_pedestals.size());
    std::for_each(m_pedestals.begin(),m_pedestals.end(),Inserter<EBDetId,Item>(m_barrel));
  }
}
