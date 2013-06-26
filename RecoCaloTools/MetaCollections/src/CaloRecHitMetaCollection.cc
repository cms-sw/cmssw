#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollection.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class CaloRecHitMetaCollectionItem {
public:
  virtual ~CaloRecHitMetaCollectionItem() {}
  virtual int find(const DetId& id) const = 0;
  virtual const CaloRecHit* at(int index) const = 0;
};

template <class T>
class CaloRecHitMetaCollectionItemT : public CaloRecHitMetaCollectionItem {
public:
  CaloRecHitMetaCollectionItemT(const T* coll, int start) : m_collection(coll),m_start(start) { }
  virtual int find(const DetId& id) const {
    typename T::const_iterator i;
    i=m_collection->find(id);
    return (i==m_collection->end())?(-1):(i-m_collection->begin()+m_start);
  }
  virtual const CaloRecHit* at(int index) const {
    return &((*m_collection)[index-m_start]);
  }
private:
  const T* m_collection;
  int m_start;
};

typedef std::multimap<int, CaloRecHitMetaCollectionItem*>::const_iterator find_iterator;

int CaloRecHitMetaCollection::findIndex(const DetId& id) const {
  return id.rawId()>>25;
}

CaloRecHitMetaCollection::const_iterator CaloRecHitMetaCollection::find(const DetId& id) const {
  std::pair<find_iterator,find_iterator> options=m_findTool.equal_range(findIndex(id));
  int pos=-1;
  for (find_iterator i=options.first; i!=options.second; i++) {
    pos=i->second->find(id);
    if (pos>=0) break;
  }
  return (pos<0)?(end()):(const_iterator(this,pos));
}

const CaloRecHit* CaloRecHitMetaCollection::at(const_iterator::offset_type i) const {
  std::map<int, CaloRecHitMetaCollectionItem*>::const_iterator q=m_items.lower_bound(i);
  return (q==m_items.end())?(0):(q->second->at(i));
}

CaloRecHitMetaCollection::~CaloRecHitMetaCollection() {
  for (std::map<int, CaloRecHitMetaCollectionItem*>::iterator i=m_items.begin(); i!=m_items.end(); i++)
    delete i->second;
}


void CaloRecHitMetaCollection::add(const HBHERecHitCollection* hbhe) {
  if (hbhe->size()==0) return; // do not add empty collections (can cause problems)
  CaloRecHitMetaCollectionItem* i=new CaloRecHitMetaCollectionItemT<HBHERecHitCollection>(hbhe,size_);
  size_+=hbhe->size();
  m_items.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(size_-1,i));
  m_findTool.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(findIndex(DetId(DetId::Hcal,HcalBarrel)),i));
  m_findTool.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(findIndex(DetId(DetId::Hcal,HcalEndcap)),i));
}

void CaloRecHitMetaCollection::add(const HORecHitCollection* ho) {
  if (ho->size()==0) return; // do not add empty collections (can cause problems)
  CaloRecHitMetaCollectionItem* i=new CaloRecHitMetaCollectionItemT<HORecHitCollection>(ho,size_);
  size_+=ho->size();
  m_items.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(size_-1,i));
  m_findTool.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(findIndex(DetId(DetId::Hcal,HcalOuter)),i));
}

void CaloRecHitMetaCollection::add(const HFRecHitCollection* hf) {
  if (hf->size()==0) return; // do not add empty collections (can cause problems)
  CaloRecHitMetaCollectionItem* i=new CaloRecHitMetaCollectionItemT<HFRecHitCollection>(hf,size_);
  size_+=hf->size();
  m_items.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(size_-1,i));
  m_findTool.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(findIndex(DetId(DetId::Hcal,HcalForward)),i));
}


void CaloRecHitMetaCollection::add(const EcalRecHitCollection* ecal) {
  if (ecal->size()==0) return; // do not add empty collections (can cause problems)
  CaloRecHitMetaCollectionItem* i=new CaloRecHitMetaCollectionItemT<EcalRecHitCollection>(ecal,size_);
  size_+=ecal->size();
  m_items.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(size_-1,i));
  m_findTool.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(findIndex(DetId(DetId::Ecal,EcalBarrel)),i));
  m_findTool.insert(std::pair<int,CaloRecHitMetaCollectionItem*>(findIndex(DetId(DetId::Ecal,EcalEndcap)),i));
}

