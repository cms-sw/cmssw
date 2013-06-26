#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionFast.h"
#include <algorithm>

CaloRecHitMetaCollectionFast::CaloRecHitMetaCollectionFast()  {
  dirty_=false;
}

void CaloRecHitMetaCollectionFast::add(const CaloRecHit* hit) {
  hits_.push_back(hit);
  dirty_=true;
  size_++;
}

class CRHMCVComp {
public:
  typedef const CaloRecHit* T;
  typedef const DetId& key_type;
  
  bool operator()(key_type a, T const& b) const { return a < b->detid(); }
  bool operator()(T const& a, key_type b) const { return a->detid() < b; }
  bool operator()(T const& a, T const& b) const { return a->detid() < b->detid(); }
};


CaloRecHitMetaCollectionV::const_iterator CaloRecHitMetaCollectionFast::find(const DetId& id) const {
  if (dirty_) sort();

  CRHMCVComp comp;

  std::vector<const CaloRecHit*>::const_iterator last=hits_.end();
  std::vector<const CaloRecHit*>::const_iterator first=hits_.begin();
  std::vector<const CaloRecHit*>::const_iterator loc =std::lower_bound(first,
								       last,
								       id,
								       comp);
  return loc == last || comp(id, *loc) ? end() : const_iterator(this,loc - hits_.begin());   
}


const CaloRecHit* CaloRecHitMetaCollectionFast::at(const_iterator::offset_type i) const {
  if (dirty_) sort();
  return ((i<0 || i>=(const_iterator::offset_type)size_)?(0):(hits_[i]));
}


void CaloRecHitMetaCollectionFast::sort() const {
  if (dirty_) {
    CRHMCVComp comp;
    std::sort(hits_.begin(),hits_.end(),comp);
    dirty_=false;
  }
}

