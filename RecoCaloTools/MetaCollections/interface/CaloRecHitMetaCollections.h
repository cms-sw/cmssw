#ifndef RECOCALOTOOLS_METACOLLECTIONS_CALORECHITMETACOLLECTIONS_H
#define RECOCALOTOOLS_METACOLLECTIONS_CALORECHITMETACOLLECTIONS_H 1

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionV.h"

template <class C> 
class SimpleCaloRecHitMetaCollection : public CaloRecHitMetaCollectionV {
public:
  typedef CaloRecHitMetaCollectionV::const_iterator const_iterator;

  SimpleCaloRecHitMetaCollection(const C& c) : coll_(c) { size_=coll_.size(); }
  SimpleCaloRecHitMetaCollection(const C* c) : coll_(*c) { size_=coll_.size(); }
  virtual const_iterator find(const DetId& id) const {
    const_iterator i=end();
    typename C::const_iterator j=coll_.find(id);
    if (j!=coll_.end()) {
      int delta=j-coll_.begin();
      i=const_iterator(this,delta);
    }
    return i;
  }
  virtual const CaloRecHit* at(const_iterator::offset_type i) const {
    return &(coll_[i]);
  }
private:
  const C& coll_;
};

typedef SimpleCaloRecHitMetaCollection<HBHERecHitCollection> HBHERecHitMetaCollection;
typedef SimpleCaloRecHitMetaCollection<HFRecHitCollection> HFRecHitMetaCollection;
typedef SimpleCaloRecHitMetaCollection<HORecHitCollection> HORecHitMetaCollection;
typedef SimpleCaloRecHitMetaCollection<EcalRecHitCollection> EcalRecHitMetaCollection;

#endif
