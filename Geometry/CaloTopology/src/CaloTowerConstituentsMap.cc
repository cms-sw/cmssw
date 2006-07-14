#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

CaloTowerConstituentsMap::CaloTowerConstituentsMap() :
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
      if (hid.subdet()==HcalBarrel && standardHB_ ||
	  hid.subdet()==HcalEndcap && standardHE_ ||
	  hid.subdet()==HcalOuter && standardHO_ ||
	  hid.subdet()==HcalForward && standardHF_) {
	if (hid.subdet()==HcalForward && hid.ietaAbs()==29)  // special handling for tower 29
	  tid=CaloTowerDetId(30*hid.zside(),hid.iphi());
	else 
	  tid=CaloTowerDetId(hid.ieta(),hid.iphi());
      }      
    } else if (id.det()==DetId::Ecal) {
      EcalSubdetector esd=(EcalSubdetector)id.subdetId();
      if (esd==EcalBarrel && standardEB_) {
	EBDetId ebid(id);
	//For the moment making trigTowerDetId consistent here
	// TODO Change soon elsewhere
	int tower_iphi=(ebid.tower_iphi()-2);
	if (tower_iphi < 1)
	  tower_iphi+=72;
	tid=CaloTowerDetId(ebid.tower_ieta(),tower_iphi);
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

  // dealing with topo dependency...
  
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
