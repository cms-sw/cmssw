#ifndef ES_COND_OBJECT_CONTAINER_HH
#define ES_COND_OBJECT_CONTAINER_HH

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/EcalDetId/interface/EcalContainer.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

template <typename T>
class ESCondObjectContainer {
public:
  typedef T Item;
  typedef Item value_type;
  typedef ESCondObjectContainer<T> self;
  typedef typename std::vector<Item> Items;
  typedef typename std::vector<Item>::const_iterator const_iterator;
  typedef typename std::vector<Item>::iterator iterator;

  ESCondObjectContainer(){};
  ~ESCondObjectContainer(){};

  inline const Items &preshowerItems() const { return es_.items(); };

  inline const Item &preshower(size_t hashedIndex) const { return es_.item(hashedIndex); }

  inline void insert(std::pair<uint32_t, Item> const &a) {
    if (DetId(a.first).subdetId() == EcalPreshower) {
      es_.insert(a);
    }
  }

  inline const_iterator find(uint32_t rawId) const { return es_.find(rawId); }

  inline const_iterator begin() const { return es_.begin(); }

  inline const_iterator end() const { return es_.end(); }

  inline void setValue(const uint32_t id, const Item &item) { (*this)[id] = item; }

  inline const self &getMap() const { return *this; }

  inline size_t size() const { return es_.size(); }
  // add coherent operator++, not needed now -- FIXME

  inline Item &operator[](uint32_t rawId) { return es_[rawId]; }

  inline Item const &operator[](uint32_t rawId) const { return es_[rawId]; }

private:
  EcalContainer<ESDetId, Item> es_;

  COND_SERIALIZABLE;
};

typedef ESCondObjectContainer<float> ESFloatCondObjectContainer;
#endif
