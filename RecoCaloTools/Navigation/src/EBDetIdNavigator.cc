#include "RecoCaloTools/Navigation/interface/EBDetIdNavigator.h"

EBDetIdNavigator::EBDetIdNavigator(const EBDetId& startingPoint) : startingPoint_(startingPoint), currentPoint_(startingPoint) {
}

void EBDetIdNavigator::home() {
  currentPoint_=startingPoint_;
}

EBDetId EBDetIdNavigator::incrementIeta() {
  if (currentPoint_.ieta()==EBDetId::MAX_IETA) currentPoint_=EBDetId(0); // null det id
  else if (currentPoint_.ieta()==-1) currentPoint_=EBDetId(1,currentPoint_.iphi());
  else currentPoint_=EBDetId(currentPoint_.ieta()+1,currentPoint_.iphi());
  return currentPoint_;
}

EBDetId EBDetIdNavigator::decrementIeta() {
  if (currentPoint_.ieta()==-EBDetId::MAX_IETA) currentPoint_=EBDetId(0); // null det id
  else if (currentPoint_.ieta()==1) currentPoint_=EBDetId(-1,currentPoint_.iphi());
  else currentPoint_=EBDetId(currentPoint_.ieta()-1,currentPoint_.iphi());
  return currentPoint_;
}

EBDetId EBDetIdNavigator::incrementIphi() {
  if (currentPoint_.iphi()==EBDetId::MAX_IPHI) currentPoint_=EBDetId(currentPoint_.ieta(),EBDetId::MIN_IPHI);
  else currentPoint_=EBDetId(currentPoint_.ieta(),currentPoint_.iphi()+1);
  return currentPoint_;
}

EBDetId EBDetIdNavigator::decrementIphi() {
  if (currentPoint_.iphi()==EBDetId::MIN_IPHI) currentPoint_=EBDetId(currentPoint_.ieta(),EBDetId::MAX_IPHI);
  else currentPoint_=EBDetId(currentPoint_.ieta(),currentPoint_.iphi()-1);
  return currentPoint_;
}
