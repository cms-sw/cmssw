#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>

EcalTrigTowerConstituentsMap::EcalTrigTowerConstituentsMap() {}

EcalTrigTowerDetId EcalTrigTowerConstituentsMap::barrelTowerOf(const DetId& id) {
  assert(id.det() == DetId::Ecal && id.subdetId() == EcalBarrel);
  //--------------------
  // Ecal Barrel
  //--------------------
  EBDetId myId(id);
  return myId.tower();
}

EcalTrigTowerDetId EcalTrigTowerConstituentsMap::towerOf(const DetId& id) const {
  if (id.det() == DetId::Ecal && id.subdetId() == EcalBarrel) {
    //--------------------
    // Ecal Barrel
    //--------------------
    EBDetId myId(id);
    return myId.tower();
  } else if (id.det() == DetId::Ecal && id.subdetId() == EcalEndcap) {
    //--------------------
    // Ecal Endcap
    //--------------------
    EEDetId originalId(id);
    // DetId wrappedId=wrapEEDetId(id);
    DetId wrappedId(originalId);
    EcalTowerMap::const_iterator i = m_items.find(wrappedId);
    if (i != m_items.end()) {
      int etaTower = i->tower.ietaAbs();
      int phiTower = i->tower.iphi();
      //trigger tower <-> crystal maping read
      //..........from file and done only for 1 quadrant
      //move from quadrant 1 to the actual one:
      // phiTower = changeTowerQuadrant(phiTower, 1, originalId.iquadrant());
      edm::LogVerbatim("EcalTrigTowerConstituentsMap")
          << "EcalTrigTowerConstituentsMap " << originalId.zside() << " " << etaTower << " " << phiTower;
      return EcalTrigTowerDetId(originalId.zside(), EcalEndcap, etaTower, phiTower);
    }
  }
  return EcalTrigTowerDetId(0);
}

DetId EcalTrigTowerConstituentsMap::wrapEEDetId(const DetId& eeid) const {
  if (!(eeid.det() == DetId::Ecal && eeid.subdetId() == EcalEndcap))
    return EEDetId(0);

  EEDetId myId(eeid);
  switch ((myId.iquadrant() - 1) % 4) {
    case 0:
      return DetId(EEDetId(myId.ix(), myId.iy(), 1, EEDetId::XYMODE));
      break;
    case 1:
      return DetId(EEDetId(101 - myId.ix(), myId.iy(), 1, EEDetId::XYMODE));
      break;
    case 2:
      return DetId(EEDetId(101 - myId.ix(), 101 - myId.iy(), 1, EEDetId::XYMODE));
      break;
    case 3:
      return DetId(EEDetId(myId.ix(), 101 - myId.iy(), 1, EEDetId::XYMODE));
      break;
    default:
      /*should never be reached*/
      edm::LogError("EcalTrigTowerConstituentsMapError") << "This should never be reached. Profound error!";
  }
  return EEDetId(0);
}

DetId EcalTrigTowerConstituentsMap::wrapEcalTrigTowerDetId(const DetId& id) const {
  EcalTrigTowerDetId etid(id);

  if (!(etid.det() == DetId::Ecal && etid.subdetId() == EcalTriggerTower && etid.subDet() == EcalEndcap))
    return EcalTrigTowerDetId(0);

  switch ((etid.iquadrant() - 1) % 4) {
    case 0:
      return DetId(EcalTrigTowerDetId(1, EcalEndcap, etid.ietaAbs(), etid.iphi()));
      break;
    case 1:
      return DetId(EcalTrigTowerDetId(
          1, EcalEndcap, etid.ietaAbs(), EcalTrigTowerDetId::kEETowersInPhiPerQuadrant * 2 - etid.iphi() + 1));
      break;
    case 2:
      return DetId(EcalTrigTowerDetId(
          1, EcalEndcap, etid.ietaAbs(), etid.iphi() - EcalTrigTowerDetId::kEETowersInPhiPerQuadrant * 2));
      break;
    case 3:
      return DetId(EcalTrigTowerDetId(
          1, EcalEndcap, etid.ietaAbs(), EcalTrigTowerDetId::kEETowersInPhiPerQuadrant * 4 - etid.iphi() + 1));
      break;
    default:
      /*should never be reached*/
      edm::LogError("EcalTrigTowerConstituentsMapError") << "This should never be reached. Profound error!";
  }
  return EcalTrigTowerDetId(0);
}

DetId EcalTrigTowerConstituentsMap::changeEEDetIdQuadrantAndZ(const DetId& fromid,
                                                              const int& toQuadrant,
                                                              const int& tozside) const {
  if (!(fromid.det() == DetId::Ecal && fromid.subdetId() == EcalEndcap))
    return EEDetId(0);

  EEDetId myId(fromid);
  int dQuadrant = toQuadrant - myId.iquadrant();
  switch (dQuadrant % 4) {
    case 0:
      return DetId(EEDetId(myId.ix(), myId.iy(), tozside, EEDetId::XYMODE));
      break;
    case 1:
      /* adjacent tower: they are symetric*/
      return DetId(EEDetId(101 - myId.ix(), myId.iy(), tozside, EEDetId::XYMODE));
      break;
    case 2:
      /* opposite quadrant: they are identical*/
      return DetId(EEDetId(101 - myId.ix(), 101 - myId.iy(), tozside, EEDetId::XYMODE));
      break;
    case 3:
      /* adjacent tower: they are symetric*/
      return DetId(EEDetId(myId.ix(), 101 - myId.iy(), tozside, EEDetId::XYMODE));
      break;
    default:
      /*should never be reached*/
      edm::LogError("EcalTrigTowerConstituentsMapError") << "This should never be reached. Profound error!";
  }
  return EEDetId(0);
}

int EcalTrigTowerConstituentsMap::changeTowerQuadrant(int phiTower, int fromQuadrant, int toQuadrant) const {
  int newPhiTower = phiTower;
  int dQuadrant = toQuadrant - fromQuadrant;

  switch (dQuadrant % 4) {
    case 0:
      newPhiTower = phiTower;
      break;
    case 1:
      /* adjacent tower: they are symetric*/
      newPhiTower = EcalTrigTowerDetId::kEETowersInPhiPerQuadrant * 2 - phiTower + 1;
      break;
    case 2:
      /* opposite quadrant: they are identical*/
      newPhiTower = phiTower + EcalTrigTowerDetId::kEETowersInPhiPerQuadrant * 2;
      break;
    case 3:
      /* adjacent tower: they are symetric*/
      newPhiTower = EcalTrigTowerDetId::kEETowersInPhiPerQuadrant * 4 - phiTower + 1;
      break;
    default:
      /*should never be reached*/
      edm::LogError("EcalTrigTowerConstituentsMapError") << "This should never be reached. Profound error!";
  }
  return newPhiTower;
}

void EcalTrigTowerConstituentsMap::assign(const DetId& cell, const EcalTrigTowerDetId& tower) {
  if (m_items.find(cell) != m_items.end()) {
    throw cms::Exception("EcalTrigTowers")
        << "Cell with id " << std::hex << cell.rawId() << std::dec << " is already mapped to a EcalTrigTower "
        << m_items.find(cell)->tower << std::endl;
  }

  m_items.insert(MapItem(cell, tower));
}

std::vector<DetId> EcalTrigTowerConstituentsMap::constituentsOf(const EcalTrigTowerDetId& id) const {
  std::vector<DetId> items;

  if (id.det() == DetId::Ecal && id.subdetId() == EcalTriggerTower && id.subDet() == EcalBarrel) {
    //--------------------
    // Ecal Barrel
    //--------------------
    // trigger towers are 5x5 crystals in the barrel
    int etaxtalMin = (id.ietaAbs() - 1) * 5 + 1;
    int phixtalMin = ((id.iphi() - 1) * 5 + 11) % 360;
    if (phixtalMin <= 0)
      phixtalMin += 360;
    int etaxtalMax = id.ietaAbs() * 5;
    int phixtalMax = ((id.iphi()) * 5 + 10) % 360;
    if (phixtalMax <= 0)
      phixtalMax += 360;
    for (int e = etaxtalMin; e <= etaxtalMax; e++)
      for (int p = phixtalMin; p <= phixtalMax; p++)
        items.emplace_back(DetId(EBDetId(id.zside() * e, p, EBDetId::ETAPHIMODE)));
  } else if (id.det() == DetId::Ecal && id.subdetId() == EcalTriggerTower && id.subDet() == EcalEndcap) {
    //--------------------
    // Ecal Endcap
    //--------------------
    //DetId myId=wrapEcalTrigTowerDetId(id);
    EcalTowerMap_by_towerDetId::const_iterator lb, ub;
    //boost::tuples::tie(lb,ub)=get<1>(m_items).equal_range(myId);
    boost::tuples::tie(lb, ub) = boost::get<1>(m_items).equal_range(id);
    while (lb != ub) {
      //EEDetId mappedId((*lb).cell);
      //items.emplace_back(changeEEDetIdQuadrantAndZ(mappedId,id.iquadrant(),id.zside()));
      items.emplace_back((*lb).cell);
      ++lb;
    }
  }

  return items;
}
