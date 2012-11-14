#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

CaloTowerConstituentsMap::CaloTowerConstituentsMap(const HcalTopology * topology) :
  m_topology(topology),
  standardHB_(false),
  standardHE_(false),
  standardHF_(false),
  standardHO_(false),
  standardEB_(false)
{
}

CaloTowerDetId CaloTowerConstituentsMap::towerOf(const DetId& id) const {
  CaloTowerDetId tid; // null to start with

  edm::SortedCollection<MapItem>::const_iterator i=m_items.find(id);
  if (i!=m_items.end()) tid=i->tower;

  if (tid.null()) {
    if (id.det()==DetId::Hcal) { 
      HcalDetId hid(id);
      if ( (hid.subdet()==HcalBarrel && standardHB_ )  ||
	   (hid.subdet()==HcalEndcap && standardHE_ )  ||
	   (hid.subdet()==HcalOuter  && standardHO_ )  ||
	   (hid.subdet()==HcalForward && standardHF_) ) {
	if ((hid.subdet()==HcalForward) && hid.ietaAbs()==29)  // special handling for tower 29
	  tid=CaloTowerDetId(30*hid.zside(),hid.iphi());
	else 
	  tid=CaloTowerDetId(hid.ieta(),hid.iphi());
      }      
    } else if (id.det()==DetId::Ecal) {
      EcalSubdetector esd=(EcalSubdetector)id.subdetId();
      if (esd==EcalBarrel && standardEB_) {
	EBDetId ebid(id);
	tid=CaloTowerDetId(ebid.tower_ieta(),ebid.tower_iphi());
      }
    }
  }

  return tid;
}

void CaloTowerConstituentsMap::assign(const DetId& cell, const CaloTowerDetId& tower) {
  if (m_items.find(cell)!=m_items.end()) {
    throw cms::Exception("CaloTowers") << "Cell with id " << std::hex << cell.rawId() << std::dec << " is already mapped to a CaloTower " << m_items.find(cell)->tower << std::endl;
  }
  m_items.push_back(MapItem(cell,tower));
}

void CaloTowerConstituentsMap::sort() {
  m_items.sort();
}

std::vector<DetId> CaloTowerConstituentsMap::constituentsOf(const CaloTowerDetId& id) const {
  std::vector<DetId> items;

  // build reverse map if needed
  if (!m_items.empty() && m_reverseItems.empty()) {
    for (edm::SortedCollection<MapItem>::const_iterator i=m_items.begin(); i!=m_items.end(); i++)
      m_reverseItems.insert(std::pair<CaloTowerDetId,DetId>(i->tower,i->cell));
  }

  /// copy from the items map
  std::multimap<CaloTowerDetId,DetId>::const_iterator j;
  std::pair<std::multimap<CaloTowerDetId,DetId>::const_iterator,std::multimap<CaloTowerDetId,DetId>::const_iterator> range=m_reverseItems.equal_range(id);
  for (j=range.first; j!=range.second; j++)
    items.push_back(j->second);

  // dealing with topo dependency...
  int nd, sd;

  if (standardHB_) {
    if (id.ietaAbs()<=m_topology->lastHBRing()) {
      m_topology->depthBinInformation(HcalBarrel,id.ietaAbs(),nd,sd);
      for (int i=0; i<nd; i++)
	items.push_back(HcalDetId(HcalBarrel,id.ieta(),id.iphi(),i+sd));
    }
  }
  if (standardHO_) {
    if (id.ietaAbs()<=m_topology->lastHORing()) {
      m_topology->depthBinInformation(HcalOuter,id.ietaAbs(),nd,sd);
      for (int i=0; i<nd; i++)
	items.push_back(HcalDetId(HcalOuter,id.ieta(),id.iphi(),i+sd));
    }
  }
  if (standardHE_) {
    if (id.ietaAbs()>=m_topology->firstHERing() && id.ietaAbs()<=m_topology->lastHERing()) {
      m_topology->depthBinInformation(HcalEndcap,id.ietaAbs(),nd,sd);
      for (int i=0; i<nd; i++)
	items.push_back(HcalDetId(HcalEndcap,id.ieta(),id.iphi(),i+sd));
    }
  }
  if (standardHF_) {
    if (id.ietaAbs()>m_topology->firstHFRing() && id.ietaAbs()<=m_topology->lastHFRing()) { 
      int ieta=id.ieta();
      m_topology->depthBinInformation(HcalForward,id.ietaAbs(),nd,sd);
      for (int i=0; i<nd; i++)
	items.push_back(HcalDetId(HcalForward,ieta,id.iphi(),i+sd));
       if (id.ietaAbs() == 30) {
       ieta = 29*id.zside();
       m_topology->depthBinInformation(HcalForward,ieta,nd,sd);
       for (int i=0; i<nd; i++)
         items.push_back(HcalDetId(HcalForward,ieta,id.iphi(),i+sd));
       }
    }
  }
  if (standardEB_ && id.ietaAbs()<=EBDetId::MAX_IETA/5) {
    HcalDetId hid(HcalBarrel,id.ieta(),id.iphi(),1); // for the limits
    int etaMin, etaMax;
    if (hid.zside() == -1) {
      etaMin = hid.crystal_ieta_high();
      etaMax = hid.crystal_ieta_low();
    } else {
      etaMin = hid.crystal_ieta_low();
      etaMax = hid.crystal_ieta_high();
    }
    for (int ie=etaMin; ie<=etaMax; ie++)
      for (int ip=hid.crystal_iphi_low(); ip<=hid.crystal_iphi_high(); ip++)
	items.push_back(EBDetId(ie,ip));
  }

  return items;
}

void CaloTowerConstituentsMap::useStandardHB(bool use) {
  standardHB_=use;
}
void CaloTowerConstituentsMap::useStandardHE(bool use) {
  standardHE_=use;
}
void CaloTowerConstituentsMap::useStandardHO(bool use) {
  standardHO_=use;
}
void CaloTowerConstituentsMap::useStandardHF(bool use) {
  standardHF_=use;
}
void CaloTowerConstituentsMap::useStandardEB(bool use) {
  standardEB_=use;
}
