#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionV.h"

CaloRecHitMetaCollectionV::CaloRecHitMetaCollectionV() : size_(0) {
}

CaloRecHitMetaCollectionV::const_iterator CaloRecHitMetaCollectionV::find(const DetId& id) const {
  const_iterator i=begin();
  const_iterator e=end();
  for (; i!=e && i->detid()!=id; i++);
  return i;
}

const CaloRecHit& CaloRecHitMetaCollectionV::const_iterator::operator*() const {
  return (*collection_->at(offset_));
}

const CaloRecHit* CaloRecHitMetaCollectionV::const_iterator::operator->() const {
  return (collection_==0)?(0):(collection_->at(offset_));
}


bool CaloRecHitMetaCollectionV::Iterator::operator==(const CaloRecHitMetaCollectionV::Iterator& it) const {
  return collection_==it.collection_ && offset_==it.offset_;
}

bool CaloRecHitMetaCollectionV::Iterator::operator!=(const CaloRecHitMetaCollectionV::Iterator& it) const {
  return collection_!=it.collection_ || offset_!=it.offset_;
}

CaloRecHitMetaCollectionV::Iterator& CaloRecHitMetaCollectionV::Iterator::operator++() {
  offset_++;
  return (*this);
}

CaloRecHitMetaCollectionV::Iterator CaloRecHitMetaCollectionV::Iterator::operator++(int) {
  Iterator tmp(*this);
  offset_++;
  return tmp;
}

CaloRecHitMetaCollectionV::Iterator& CaloRecHitMetaCollectionV::Iterator::operator--() {
  offset_--;
  return (*this);
}


CaloRecHitMetaCollectionV::Iterator CaloRecHitMetaCollectionV::Iterator::operator--(int) {
  Iterator tmp(*this);
  offset_--;
  return tmp;
}

CaloRecHitMetaCollectionV::Iterator::reference CaloRecHitMetaCollectionV::Iterator::operator[](const difference_type n) const {
  return *(collection_->at(offset_+n));
}

CaloRecHitMetaCollectionV::Iterator& CaloRecHitMetaCollectionV::Iterator::operator+=(const CaloRecHitMetaCollectionV::Iterator::difference_type n) {
  offset_+=n;
  return (*this);
}

CaloRecHitMetaCollectionV::Iterator CaloRecHitMetaCollectionV::Iterator::operator+(const CaloRecHitMetaCollectionV::Iterator::difference_type n) const {
  return Iterator(collection_,offset_+n);
}

CaloRecHitMetaCollectionV::Iterator& CaloRecHitMetaCollectionV::Iterator::operator-=(const CaloRecHitMetaCollectionV::Iterator::difference_type n) {
  offset_-=n;
  return (*this);
}

CaloRecHitMetaCollectionV::Iterator CaloRecHitMetaCollectionV::Iterator::operator-(const CaloRecHitMetaCollectionV::Iterator::difference_type n) const {
  return Iterator(collection_,offset_-n);
}

bool CaloRecHitMetaCollectionV::Iterator::operator<(const CaloRecHitMetaCollectionV::Iterator& i) const {
  return offset_<i.offset_;
}
