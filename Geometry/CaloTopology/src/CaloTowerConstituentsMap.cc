#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"

#include <memory>

//#define EDM_ML_DEBUG

CaloTowerConstituentsMap::~CaloTowerConstituentsMap() {
  delete m_reverseItems.load();
  m_reverseItems = nullptr;
}
CaloTowerConstituentsMap::CaloTowerConstituentsMap(const HcalTopology * hcaltopo, const CaloTowerTopology * cttopo) :
  m_hcaltopo(hcaltopo),
  m_cttopo(cttopo),
  standardHB_(false),
  standardHE_(false),
  standardHF_(false),
  standardHO_(false),
  standardEB_(false),
  m_reverseItems(nullptr)
{
}

CaloTowerDetId CaloTowerConstituentsMap::towerOf(const DetId& id) const {
  CaloTowerDetId tid; // null to start with

  edm::SortedCollection<MapItem>::const_iterator i=m_items.find(id);
  if (i!=m_items.end()) tid=i->tower;

  //use hcaltopo when dealing with hcal detids
  if (tid.null()) {
    if (id.det()==DetId::Hcal) { 
      HcalDetId hid(id);
      if ( (hid.subdet()==HcalBarrel && standardHB_ )  ||
	   (hid.subdet()==HcalEndcap && standardHE_ )  ||
	   (hid.subdet()==HcalOuter  && standardHO_ )  ||
	   (hid.subdet()==HcalForward && standardHF_) ) {
        tid = CaloTowerDetId(m_cttopo->convertHcaltoCT(hid.ietaAbs(),hid.subdet())*hid.zside(),hid.iphi());
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
  m_items.emplace_back(MapItem(cell,tower));
}

void CaloTowerConstituentsMap::sort() {
  m_items.sort();
  
//  for (auto const & it : m_items)
//    std::cout << std::hex << it.cell.rawId() << " " << it.tower.rawId() << std::dec << std::endl;

}

std::vector<DetId> CaloTowerConstituentsMap::constituentsOf(const CaloTowerDetId& id) const {
#ifdef EDM_ML_DEBUG
  std::cout << "Get constituent of " << std::hex << id.rawId() << std::dec
	    << " ID " << id << " ieta " << id.ieta() << std::endl;
#endif
  std::vector<DetId> items;
  if (id.ieta() == 0) return items;

  // build reverse map if needed
  if(!m_reverseItems.load(std::memory_order_acquire)) {
      std::unique_ptr<std::multimap<CaloTowerDetId,DetId>> ptr{new std::multimap<CaloTowerDetId,DetId>};
      for (auto m_item : m_items)
          ptr->insert(std::pair<CaloTowerDetId,DetId>(m_item.tower,m_item.cell));
      std::multimap<CaloTowerDetId,DetId>* expected = nullptr;
      if(m_reverseItems.compare_exchange_strong(expected, ptr.get(), std::memory_order_acq_rel)) {
          ptr.release();
      }
  }

  /// copy from the items map
  std::multimap<CaloTowerDetId,DetId>::const_iterator j;
  auto range=(*m_reverseItems.load(std::memory_order_acquire)).equal_range(id);
  for (j=range.first; j!=range.second; j++)
    items.emplace_back(j->second);

  // dealing with topo dependency...
  //use cttopo when dealing with calotower detids
  int nd, sd;
  int hcal_ieta = m_cttopo->convertCTtoHcal(id.ietaAbs());
  
  if (standardHB_) {
    if (id.ietaAbs()<=m_cttopo->lastHBRing()) {
      m_hcaltopo->depthBinInformation(HcalBarrel,hcal_ieta,id.iphi(),id.zside(),nd,sd);
      for (int i=0; i<nd; i++) {
	if (m_hcaltopo->getMergePositionFlag()) {
	  HcalDetId hid = m_hcaltopo->mergedDepthDetId(HcalDetId(HcalBarrel,hcal_ieta*id.zside(),id.iphi(),i+sd));
	  if (std::find(items.begin(),items.end(),hid) == items.end()) {
	    items.emplace_back(hid);
#ifdef EDM_ML_DEBUG
	    std::cout << id << " Depth " << i << ":" << i+sd << " " << hid <<"\n";
#endif
	  }
	} else {
	  HcalDetId hid(HcalBarrel,hcal_ieta*id.zside(),id.iphi(),i+sd);
	  items.emplace_back(hid);
#ifdef EDM_ML_DEBUG
	  std::cout << id << " Depth " << i << ":" << i+sd << " " << hid <<"\n";
#endif
	}
      }
    }
  }
  if (standardHO_) {
    if (id.ietaAbs()<=m_cttopo->lastHORing()) {
      m_hcaltopo->depthBinInformation(HcalOuter,hcal_ieta,id.iphi(),id.zside(),nd,sd);
      for (int i=0; i<nd; i++) {
	HcalDetId hid(HcalOuter,hcal_ieta*id.zside(),id.iphi(),i+sd);
        items.emplace_back(hid);
#ifdef EDM_ML_DEBUG
	std::cout << id << " Depth " << i << ":" << i+sd << " " << hid <<"\n";
#endif
      }
    }
  }
  if (standardHE_) {
    if (id.ietaAbs()>=m_cttopo->firstHERing() && id.ietaAbs()<=m_cttopo->lastHERing()) {
      m_hcaltopo->depthBinInformation(HcalEndcap,hcal_ieta,id.iphi(),id.zside(),nd,sd);
      for (int i=0; i<nd; i++) {
	if (m_hcaltopo->getMergePositionFlag()) {
	  HcalDetId hid = m_hcaltopo->mergedDepthDetId(HcalDetId(HcalEndcap,hcal_ieta*id.zside(),id.iphi(),i+sd));
	  if (std::find(items.begin(),items.end(),hid) == items.end()) {
	    items.emplace_back(hid);
#ifdef EDM_ML_DEBUG
	    std::cout << id << " Depth " << i << ":" << i+sd << " " << hid <<"\n";
#endif
	  } 
	} else {
	  HcalDetId hid(HcalEndcap,hcal_ieta*id.zside(),id.iphi(),i+sd);
	  items.emplace_back(hid);
#ifdef EDM_ML_DEBUG
	  std::cout << id << " Depth " << i << ":" << i+sd << " " << hid <<"\n";
#endif
	}
      }
    }
  }
  if (standardHF_) {
    if (id.ietaAbs()>=m_cttopo->firstHFRing() && id.ietaAbs()<=m_cttopo->lastHFRing()) { 
      m_hcaltopo->depthBinInformation(HcalForward,hcal_ieta,id.iphi(),id.zside(),nd,sd);
      for (int i=0; i<nd; i++) {
	HcalDetId hid(HcalForward,hcal_ieta*id.zside(),id.iphi(),i+sd);
        items.emplace_back(hid);
#ifdef EDM_ML_DEBUG
	std::cout << id << " Depth " << i << ":" << i+sd << " " << hid <<"\n";
#endif
      }
      // special handling for first HF tower
      if (id.ietaAbs() == m_cttopo->firstHFRing()) {
        int hcal_ieta2 = hcal_ieta-1;
        m_hcaltopo->depthBinInformation(HcalForward,hcal_ieta2,id.iphi(),id.zside(),nd,sd);
        for (int i=0; i<nd; i++) {
	  HcalDetId hid(HcalForward,hcal_ieta2*id.zside(),id.iphi(),i+sd);
          items.emplace_back(hid);
#ifdef EDM_ML_DEBUG
	  std::cout << id << " Depth " << i << ":" << i+sd << " " << hid <<"\n";
#endif
	}
      }
    }
  }
  if (standardEB_ && hcal_ieta<=EBDetId::MAX_IETA/5) {
    HcalDetId hid(HcalBarrel,hcal_ieta*id.zside(),id.iphi(),1); // for the limits
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
        items.emplace_back(EBDetId(ie,ip));
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
