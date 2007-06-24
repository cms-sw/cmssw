#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include <algorithm>

EcalPedestals::Item::Zero EcalPedestals::Item::zero;

EcalPedestals::EcalPedestals(){}
EcalPedestals::~EcalPedestals(){}

namespace {
  template<typename DetId, typename Item>
  class Inserter {
  public:
    Inserter(std::vector<Item> & iv) : v(iv){}

    void operator()(std::pair<uint32_t, const Item> const & p) {
      DetId id(p.first);
      if (id.null() || id.det()!=DetId::Ecal || id.subdetId()!=EcalBarrel ) 
	return;
      v.at(id.hashedIndex()) = p.second;
    }

    std::vector<Item> & v;
  };

  template<typename Item>
  class EcalCalibInserter {
  public:
    EcalCalibInserter(std::vector<Item> & ib, std::vector<Item> & ie ) : b(ib),e(ie){}

    void operator()(std::pair<uint32_t, const Item> const & p) {
      ::DetId id(p.first);
      if (id.null() || id.det()!=::DetId::Ecal) return;
      switch (id.subdetId()) {
      case EcalBarrel :
	{ 
	  EBDetId ib(p.first);
	  b.at(ib.hashedIndex()) = p.second;
	}
	break;
      case EcalEndcap :
	{ 
	EEDetId ie(p.first);
	// e.at(ie.hashedIndex()) = p.second;
	}
        break;
      default:
        return;
      }
    }

    std::vector<Item> & b;
    std::vector<Item> & e;
  };
}

Item const & EcalPedestals::operator()(DetId id) const {
  static Item dummy;
  switch (id.subdetId()) {
  case EcalBarrel :
    { 
      EBDetId ib(id);
      return barrel(ib.hashedIndex());
    }
    break;
  case EcalEndcap :
    { 
      // EEDetId ie(id);
      const_cast<EcalPedestals*>(this)->m_pedestals[id];
    }
    break;
  default:
    return dummy;
  }
  return dummy;
}



void EcalPedestals::update() const {
  if (m_barrel.empty()) {
    // FIXME
    m_barrel.resize(m_pedestals.size());
    std::for_each(m_pedestals.begin(),m_pedestals.end(),
		  EcalCalibInserter<Item>(m_barrel,m_endcap));
  }
}
