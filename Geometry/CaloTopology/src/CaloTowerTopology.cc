#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"


bool CaloTowerTopology::valid(const DetId& id) const {
  CaloTowerDetId tid(id);
  bool bad=(tid.ieta()==0 || tid.iphi()<=0 || tid.iphi()>72 || tid.ieta()<-41 || tid.ieta()>41);
  return !bad;
}

std::vector<DetId> CaloTowerTopology::east(const DetId& id) const {
  std::vector<DetId> dd;
  CaloTowerDetId tid(id);
  int ieta=tid.ieta()-1;
  if (ieta==0) ieta--;
  if (ieta>=-41) 
    dd.push_back(CaloTowerDetId(ieta,tid.iphi()));
  return dd;
}

std::vector<DetId> CaloTowerTopology::west(const DetId& id) const {
  std::vector<DetId> dd;
  CaloTowerDetId tid(id);
  int ieta=tid.ieta()+1;
  if (ieta==0) ieta++;
  if (ieta<=41) 
    dd.push_back(CaloTowerDetId(ieta,tid.iphi()));
  return dd;
}

std::vector<DetId> CaloTowerTopology::north(const DetId& id) const {
  CaloTowerDetId tid(id);
  int iphi_n=tid.iphi()+1;
  if (iphi_n>72) iphi_n=1;
  std::vector<DetId> dd;
  dd.push_back(CaloTowerDetId(tid.ieta(),iphi_n));
  return dd;
}

std::vector<DetId> CaloTowerTopology::south(const DetId& id) const {
  CaloTowerDetId tid(id);
  int iphi_s=tid.iphi()-1;
  if (iphi_s==0) iphi_s=72;
  std::vector<DetId> dd;
  dd.push_back(CaloTowerDetId(tid.ieta(),iphi_s));
  return dd;
}

std::vector<DetId> CaloTowerTopology::up(const DetId& id) const {
  return std::vector<DetId>();
}

std::vector<DetId> CaloTowerTopology::down(const DetId& id) const {
  return std::vector<DetId>();
}

