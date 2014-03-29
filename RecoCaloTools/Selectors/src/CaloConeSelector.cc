#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"


template <class T>
CaloConeSelector<T>::CaloConeSelector(double dR, const CaloGeometry* geom) :
  geom_(geom),deltaR_(dR),detector_(DetId::Detector(0)),subdet_(0) {
}

template <class T>
CaloConeSelector<T>::CaloConeSelector(double dR, const CaloGeometry* geom, DetId::Detector detector, int subdet) : 
  geom_(geom),deltaR_(dR),detector_(detector),subdet_(subdet) {
}

template <class T>
std::auto_ptr<edm::SortedCollection<T> > CaloConeSelector<T>::select(double eta, double phi, const edm::SortedCollection<T>& inputCollection) {
  GlobalPoint p(GlobalPoint::Cylindrical(1,phi,tanh(eta)));
  return select(p,inputCollection);
}

template <class T>
std::auto_ptr<edm::SortedCollection<T> > CaloConeSelector<T>::select(const GlobalPoint& p, const edm::SortedCollection<T>& inputCollection) {
  edm::SortedCollection<T> *c = new edm::SortedCollection<T>();

  this->selectCallback(p, inputCollection, [&](const T& n) {
    c->push_back(n);
  });

  c->sort();
  return std::auto_ptr<edm::SortedCollection<T> >(c);
}

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
template class CaloConeSelector<HBHERecHit>;
template class CaloConeSelector<HFRecHit>;
template class CaloConeSelector<HORecHit>;
template class CaloConeSelector<EcalRecHit>;
