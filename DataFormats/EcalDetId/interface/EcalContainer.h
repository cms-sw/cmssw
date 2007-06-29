#ifndef ECALDETID_ECALCONTAINER_H
#define ECALDETID_ECALCONTAINER_H

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include <vector>
#include <utility>
#include <algorithm>


/* a generic container for ecal items
 * stores them in two vectors: one for barrel, one for endcap
 * provides access by hashedIndex and by DetId...
 */

namespace detailsEcalContainer{
  template<typename DetId, typename Item>
  class SingleInserter {
  public:
    SingleInserter(std::vector<Item> & iv) : v(iv){}
    
    void operator()(std::pair<uint32_t, const Item> const & p) {
      DetId id(p.first);
      if (id.null() || id.det()!=DetId::Ecal || id.subdetId()!=DetId::Subdet ) 
	return;
      v.at(id.hashedIndex()) = p.second;
    }
    
    std::vector<Item> & v;
  };
  
  template<typename Item>
  class Inserter {
  public:
    Inserter(std::vector<Item> & ib, std::vector<Item> & ie ) : b(ib),e(ie){}

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
	  e.at(ie.hashedIndex()) = p.second;
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



template<typename T>
clall EcalContainer {
  typedef EcalContainer<T> self;
  typedef T Item;
  typedef Item value_type;
  
  template<typename Iter>
    void load(Iter b, Iter e) const {
    const_cast<self&>(*this).load(b,e);
  }
  
  template<typename Iter>
    void load(Iter b, Iter e) const {
    if (m_barrel.empty()) {
      m_barrel.resize(EBDetId::SIZE_HASH);
      m_endcap.resize(EEDetId::SIZE_HASH);     
      std::for_each(b,e,
		    detailsEcalContainer::Inserter<Item>(m_barrel,m_endcap)
		    );
    }
  }
  
  Item const & barrel(size_t hashid) const {
    return m_barrel[hashid];
  }
  Item const & endcap(size_t hashid) const {
    return m_endcap[hashid];
  }

  Item const & operator()(::DetId id) const {
    static Item dummy;
    switch (id.subdetId()) {
    case EcalBarrel :
      { 
	EBDetId ib(id.rawId());
	return m_barrel[ib.hashedIndex()];
      }
      break;
    case EcalEndcap :
      { 
	EEDetId ie(id.rawId());
	return m_endcap[ie.hashedIndex()];
      }
      break;
    default:
      // FIXME (add throw)
      return dummy;
    }
    // make compiler happy
    return dummy;
  }


 private:

  std::vector<Item> m_barrel;
  std::vector<Item> m_endcap;


};


#endif // ECALCONTAINER
