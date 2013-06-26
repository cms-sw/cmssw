#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"

static const int DoubleHE = 21;
static const int QuadHF = 40;

bool CaloTowerTopology::valid(const DetId& id) const {
  CaloTowerDetId tid(id);
  bool bad=(tid.ieta()==0 || tid.iphi()<=0 || tid.iphi()>72 || tid.ieta()<-41 || tid.ieta()>41);
  return !bad;
}

std::vector<DetId> CaloTowerTopology::east(const DetId& id) const {
  std::vector<DetId> dd;
  CaloTowerDetId tid(id);
  int ieta=tid.ieta();
  int iphi=tid.iphi();

  if (ieta==1) {
    ieta=-1;
  } else if (ieta==DoubleHE) {
    ieta--;
    dd.push_back(CaloTowerDetId(ieta,iphi+1));    
  } else if (ieta-1==-DoubleHE) {
    if ((iphi%2)==0) iphi--;
    ieta--;
  } else if (ieta==QuadHF) {
    ieta--;
    dd.push_back(CaloTowerDetId(ieta,((iphi+1)%72)+1));    
  } else if (ieta-1==-QuadHF) {
    if (((iphi-1)%4)==0) {
      if (iphi==1) iphi=71;
      else         iphi-=2;
    }
    ieta--;
  } else {
    ieta--;
  }

  if (ieta>=-41) dd.push_back(CaloTowerDetId(ieta,iphi));
  return dd;
}

std::vector<DetId> CaloTowerTopology::west(const DetId& id) const {
  std::vector<DetId> dd;
  CaloTowerDetId tid(id);

  int ieta=tid.ieta();
  int iphi=tid.iphi();

  if (ieta==-1) {
    ieta=1;
  } else if (ieta==-DoubleHE) {
    ieta++;
    dd.push_back(CaloTowerDetId(ieta,iphi+1));    
  } else if (ieta+1==DoubleHE) {
    if ((iphi%2)==0) iphi--;
    ieta++;
  } else if (ieta==-QuadHF) {
    ieta++;
    dd.push_back(CaloTowerDetId(ieta,((iphi+1)%72)+1));    
  } else if (ieta+1==QuadHF) {
    if (((iphi-1)%4)==0) {
      if (iphi==1) iphi=71;
      else         iphi-=2;
    }
    ieta++;
  } else {
    ieta++;
  }

  if (ieta<=41) dd.push_back(CaloTowerDetId(ieta,iphi));

  return dd;
}

std::vector<DetId> CaloTowerTopology::north(const DetId& id) const {
  CaloTowerDetId tid(id);
  int iphi_n=tid.iphi()+1;
  if (iphi_n>72) iphi_n=1;
  if (tid.ietaAbs()>=QuadHF) {
    iphi_n+=3;
    if (iphi_n>72) iphi_n-=72;
  } else if (tid.ietaAbs()>=DoubleHE && (iphi_n%2)==0) {
    iphi_n++;
    if (iphi_n>72) iphi_n-=72;
  }

  std::vector<DetId> dd;
  dd.push_back(CaloTowerDetId(tid.ieta(),iphi_n));
  return dd;
}

std::vector<DetId> CaloTowerTopology::south(const DetId& id) const {
  CaloTowerDetId tid(id);
  int iphi_s=tid.iphi()-1;
  if (iphi_s==0) iphi_s=72;
  if (tid.ietaAbs()>=QuadHF) {
    iphi_s-=3;
    if (iphi_s<=0) iphi_s+=72;
  } else if (tid.ietaAbs()>=DoubleHE && (iphi_s%2)==0) {
    iphi_s--;
  }

  std::vector<DetId> dd;
  dd.push_back(CaloTowerDetId(tid.ieta(),iphi_s));
  return dd;
}

std::vector<DetId> CaloTowerTopology::up(const DetId& /*id*/) const {
  return std::vector<DetId>();
}

std::vector<DetId> CaloTowerTopology::down(const DetId& /*id*/) const {
  return std::vector<DetId>();
}

