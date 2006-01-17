#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollection.h"
#include <iostream>

HBHERecHitCollection hbhe1,hbhe2;
HORecHitCollection ho;
HFRecHitCollection hf;

int main() {

  /// create collections of rechits
  for (int ieta=1; ieta<5; ieta++) {
    int iphi=1;
    int depth=1;
    hbhe1.push_back(HBHERecHit(HcalDetId(HcalBarrel,ieta,iphi,depth),ieta,iphi));
    hbhe2.push_back(HBHERecHit(HcalDetId(HcalEndcap,ieta+20,iphi,depth),ieta,iphi));
    ho.push_back(HORecHit(HcalDetId(HcalOuter,ieta+4,iphi,4),ieta*4,3));
    hf.push_back(HFRecHit(HcalDetId(HcalForward,ieta+31,iphi,1),ieta*29,4));
  }

  CaloRecHitMetaCollection f;
  f.add(&hbhe1);
  f.add(&hbhe2);
  f.add(&ho);
  f.add(&hf);

  for (CaloRecHitMetaCollection::const_iterator j=f.begin(); j!=f.end(); j++) {
    std::cout << (*j) << std::endl;
  }

  HcalDetId id;

  id=HcalDetId(HcalBarrel,1,1,1);
  if (f.find(id)==f.end()) std::cout << "Did not find " << id << std::endl;
  else std::cout << "Found : " << *(f.find(id)) << " for " << id << std::endl;

  id=HcalDetId(HcalOuter,5,1,4);
  if (f.find(id)==f.end()) std::cout << "Did not find " << id << std::endl;
  else std::cout << "Found : " << *(f.find(id)) << " for " << id << std::endl;

  id=HcalDetId(HcalForward,32,1,1);
  if (f.find(id)==f.end()) std::cout << "Did not find " << id << std::endl;
  else std::cout << "Found : " << *(f.find(id)) << " for " << id << std::endl;

  id=HcalDetId(HcalEndcap,21,1,1);
  if (f.find(id)==f.end()) std::cout << "Did not find " << id << std::endl;
  else std::cout << "Found : " << *(f.find(id)) << " for " << id << std::endl;

  id=HcalDetId(HcalBarrel,-1,1,1);
  if (f.find(id)==f.end()) std::cout << "Did not find " << id << std::endl;
  else std::cout << "Found : " << *(f.find(id)) << " for " << id << std::endl;

}
