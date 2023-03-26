// -*- Mode: C++; c-basic-offset: 8;  -*-
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>
#include <sstream>

using boost::multi_index_container;
using namespace boost::multi_index;

// -----------------------------------------------------------------------
//
// -- Conventions :
//
//    DCCid and TCCid numbering : cf slides of Ph. Gras :
//     in EE- : DCCid between 1 and 8.
//              DCCid number 1 covers the range -30 deg < phi < 10 deg.
//     in EB- : DCCid between 10 and 27.
//	        DCCid number 10 covers the range -10 deg < phi < 10 deg.
//     in EB+:  DCCid between 28 and 45.
//              DCCid number 28 covers the range -10 deg < phi < 10 deg.
//     in EE+:  DCCid between 46 and 54;
// 	        DCCid number 46 covers the range -30 deg < phi < 10 deg.
//
//    SMid : 1-18 correspond to EB+   (SMid 1 corresponds to DCC 28)
//           19-36 correspond to EB-
//
// ------------------------------------------------------------------------

EcalElectronicsMapping::EcalElectronicsMapping() {
  // Fill the map (Barrel) for the Laser Monitoring readout numbers :
  // Each DCC actually corresponds to 2 LMs,  ilm and ilm + 1

	int ilm = MIN_LM_EBM;
  for (int dcc = MIN_DCCID_EBM; dcc <= MAX_DCCID_EBM; dcc++) {
    LaserMonitoringMap_EB[dcc] = ilm;
    ilm += 2;
  }
  ilm = MIN_LM_EBP;
  for (int dcc = MIN_DCCID_EBP; dcc <= MAX_DCCID_EBP; dcc++) {
    LaserMonitoringMap_EB[dcc] = ilm;
    ilm += 2;
  }

  // Fill the map (EndCap) for the Laser Monitoring readout numbers :
  // Each DCC corresponds to onr LM, but DCC 8 (LM 80 and 81) and DCC 53 (LM 90 and 91)

  ilm = MIN_LM_EEM;
  for (int dcc = MIN_DCCID_EEM; dcc <= MAX_DCCID_EEM; dcc++) {
    LaserMonitoringMap_EE[dcc] = ilm;
    ilm += 1;
    if (dcc == 8)
      ilm += 1;
  }
  ilm = MIN_LM_EEP;
  for (int dcc = MIN_DCCID_EEP; dcc <= MAX_DCCID_EEP; dcc++) {
    LaserMonitoringMap_EE[dcc] = ilm;
    ilm += 1;
    if (dcc == 53)
      ilm += 1;
  }
}

int EcalElectronicsMapping::DCCid(const EBDetId& id) const

// SM id, between 1 (phi = -10 deg) and 18 in EB+
// between 19 (phi = -10 deg) and 27 in EB-.
// returns DCCid, currently between 10 and 27 (EB-), 28 and 45 (EB+).
// For the EE case, use getElectronicsId.
{
  int dcc = id.ism();
  if (id.zside() < 0) {
    dcc += DCCID_PHI0_EBM - 19;
  } else {
    dcc += DCCID_PHI0_EBP - 1;
  }
  return dcc;
}

int EcalElectronicsMapping::TCCid(const EBDetId& id) const

// SM id, between 1 (phi = -10 deg) and 18 in EB+
// between 19 (phi = -10 deg) and 27 in EB-.
// returns TCCid, currently between 37 and 54 (EB-), 55 and 72 (EB+).
// For the EE case, use getEcalTriggerElectronicsId.
{
  int tcc = id.ism();
  if (id.zside() < 0) {
    tcc += TCCID_PHI0_EBM - 19;
  } else {
    tcc += TCCID_PHI0_EBP - 1;
  }
  return tcc;
}

int EcalElectronicsMapping::iTT(const EcalTrigTowerDetId& id) const

// returns the index of a Trigger Tower within its TCC.
// This is between 1 and 68 for the Barrel, and between
// 1 and 32 to 34 (t.b.c.) for the EndCap.

{
  if (id.subDet() == EcalBarrel) {
    int ie = id.ietaAbs() - 1;
    int ip;
    int phi = id.iphi();
    phi += 2;
    if (phi > 72)
      phi = phi - 72;
    if (id.zside() < 0) {
      ip = ((phi - 1) % kEBTowersInPhi) + 1;
    } else {
      ip = kEBTowersInPhi - ((phi - 1) % kEBTowersInPhi);
    }

    return (ie * kEBTowersInPhi) + ip;
  } else if (id.subDet() == EcalEndcap) {
    int ie = id.ietaAbs();
    bool inner = (ie >= iEEEtaMinInner);
    if (inner) {
      ie = ie - iEEEtaMinInner;
      ie = ie % kEETowersInEtaPerInnerTCC;
    } else {
      ie = ie - iEEEtaMinOuter;
      ie = ie % kEETowersInEtaPerOuterTCC;
    }

    int ip = id.iphi();
    ip = (ip + 1) % (kEETowersInPhiPerQuadrant * 4);
    // now iphi between 0 and 71,
    // with iphi=0,1,2,3 in 1st Phi sector
    ip = ip % kEETowersInPhiPerTCC;
    int itt = kEETowersInPhiPerTCC * ie + ip + 1;
    return itt;
  } else {
    throw cms::Exception("InvalidDetId") << " Wrong EcalTrigTowerDetId in EcalElectronicsMapping::iTT. ";
    return 0;
  }
}

int EcalElectronicsMapping::TCCid(const EcalTrigTowerDetId& id) const {
  if (id.subDet() == EcalBarrel) {
    int phi = id.iphi() + 2;
    if (phi > 72)
      phi = phi - 72;
    int tcc = (phi - 1) / kEBTowersInPhi + 1;
    if (id.zside() < 0)
      tcc += 18;  // now id is the SMid
    if (id.zside() < 0) {
      tcc += TCCID_PHI0_EBM - 19;
    } else {
      tcc += TCCID_PHI0_EBP - 1;
    }
    return tcc;
  }

  else if (id.subDet() == EcalEndcap) {
    int ie = id.ietaAbs();
    bool inner = (ie >= iEEEtaMinInner);
    int ip = id.iphi();  // iphi = 1 corresponds to 0 < phi < 5 degrees
    ip = (ip + 1) % (kEETowersInPhiPerQuadrant * 4);
    // now iphi between 0 and 71,
    // with iphi=0,1,2,3 in 1st Phi sector
    int Phiindex = ip / 4;
    if (inner) {
      if (id.ieta() > 0)
        Phiindex += TCCID_PHI0_EEP_IN;
      else
        Phiindex += TCCID_PHI0_EEM_IN;
    } else {
      if (id.ieta() > 0)
        Phiindex += TCCID_PHI0_EEP_OUT;
      else
        Phiindex += TCCID_PHI0_EEM_OUT;
    }
    return Phiindex;
  } else {
    throw cms::Exception("InvalidDetId") << " Wrong EcalTrigTowerDetId in EcalElectronicsMapping::TCCid.";
    return 0;
  }
}

int EcalElectronicsMapping::DCCid(const EcalTrigTowerDetId& id) const {
  // This is needed for digitoraw. For a given Trigger Tower,
  // one needs to know to which FED it gets written.

  if (id.subDet() == EcalBarrel) {
    int phi = id.iphi() + 2;
    if (phi > 72)
      phi = phi - 72;
    int dcc = (phi - 1) / kEBTowersInPhi + 1;
    if (id.zside() < 0)
      dcc += 18;  // now id is the SMid
    if (id.zside() < 0) {
      dcc += DCCID_PHI0_EBM - 19;
    } else {
      dcc += DCCID_PHI0_EBP - 1;
    }
    return dcc;
  } else if (id.subDet() == EcalEndcap) {  //FIXME :  yes I need to improve this part of the code...
    int tccid = TCCid(id);
    int dcc = 0;
    int offset = 0;
    if (tccid >= 73) {
      tccid = tccid - 72;
      offset = 45;
    }
    if (tccid == 24 || tccid == 25 || tccid == 6 || tccid == 7)
      dcc = 4;
    if (tccid == 26 || tccid == 27 || tccid == 8 || tccid == 9)
      dcc = 5;
    if (tccid == 28 || tccid == 29 || tccid == 10 || tccid == 11)
      dcc = 6;
    if (tccid == 30 || tccid == 31 || tccid == 12 || tccid == 13)
      dcc = 7;
    if (tccid == 32 || tccid == 33 || tccid == 14 || tccid == 15)
      dcc = 8;
    if (tccid == 34 || tccid == 35 || tccid == 16 || tccid == 17)
      dcc = 9;
    if (tccid == 36 || tccid == 19 || tccid == 18 || tccid == 1)
      dcc = 1;
    if (tccid == 20 || tccid == 21 || tccid == 2 || tccid == 3)
      dcc = 2;
    if (tccid == 22 || tccid == 23 || tccid == 4 || tccid == 5)
      dcc = 3;
    dcc += offset;
    return dcc;
  } else {
    throw cms::Exception("InvalidDetId") << " Wrong EcalTrigTowerDetId in EcalElectronicsMapping::DCCid.";
    return 0;
  }
}

EcalTrigTowerDetId EcalElectronicsMapping::getTrigTowerDetId(int TCCid, int iTT) const {
  // needed for unpacking code.

  EcalSubdetector sub = subdet(TCCid, TCCMODE);
  int zIndex = zside(TCCid, TCCMODE);

  if (sub == EcalBarrel) {
    int DCCid = 0;
    int jtower = iTT - 1;
    if (zIndex > 0)
      DCCid = TCCid - TCCID_PHI0_EBP + DCCID_PHI0_EBP;
    else
      DCCid = TCCid - TCCID_PHI0_EBM + DCCID_PHI0_EBM;
    int SMid = (zIndex > 0) ? DCCid - 27 : DCCid + 9;

    int etaTT = jtower / kTowersInPhi + 1;  // between 1 and 17
    int phiTT;

    if (zIndex > 0)
      phiTT = (SMid - 1) * kTowersInPhi + (kTowersInPhi - (jtower % kTowersInPhi)) - 1;
    else
      phiTT = (SMid - 19) * kTowersInPhi + jtower % kTowersInPhi;
    phiTT++;
    phiTT = phiTT - 2;
    if (phiTT <= 0)
      phiTT = 72 + phiTT;
    EcalTrigTowerDetId tdetid(zIndex, EcalBarrel, etaTT, phiTT, EcalTrigTowerDetId::SUBDETIJMODE);
    return tdetid;
  }

  else if (sub == EcalEndcap) {
    bool EEminus = (zIndex < 0);
    bool EEplus = (zIndex > 0);
    if ((!EEminus) && (!EEplus))
      throw cms::Exception("InvalidDetId") << "EcalElectronicsMapping:  Cannot create EcalTrigTowerDetId object. ";
    int iz = 0;
    int tcc = TCCid;
    if (tcc < TCCID_PHI0_EEM_OUT + kTCCinPhi)
      iz = -1;
    else if (tcc >= TCCID_PHI0_EEP_OUT)
      iz = +1;

    bool inner = false;
    if (iz < 0 && tcc >= TCCID_PHI0_EEM_IN && tcc < TCCID_PHI0_EEM_IN + kTCCinPhi)
      inner = true;
    if (iz > 0 && tcc >= TCCID_PHI0_EEP_IN && tcc < TCCID_PHI0_EEP_IN + kTCCinPhi)
      inner = true;
    bool outer = !inner;

    int ieta = (iTT - 1) / kEETowersInPhiPerTCC;
    int iphi = (iTT - 1) % kEETowersInPhiPerTCC;
    if (inner)
      ieta += iEEEtaMinInner;
    else
      ieta += iEEEtaMinOuter;
    if (iz < 0)
      ieta = -ieta;

    int TCC_origin = 0;
    if (inner && iz < 0)
      TCC_origin = TCCID_PHI0_EEM_IN;
    if (outer && iz < 0)
      TCC_origin = TCCID_PHI0_EEM_OUT;
    if (inner && iz > 0)
      TCC_origin = TCCID_PHI0_EEP_IN;
    if (outer && iz > 0)
      TCC_origin = TCCID_PHI0_EEP_OUT;
    tcc = tcc - TCC_origin;

    iphi += kEETowersInPhiPerTCC * tcc;
    iphi = (iphi - 2 + 4 * kEETowersInPhiPerQuadrant) % (4 * kEETowersInPhiPerQuadrant) + 1;

    int tower_i = abs(ieta);
    int tower_j = iphi;

    EcalTrigTowerDetId tdetid(zIndex, EcalEndcap, tower_i, tower_j, EcalTrigTowerDetId::SUBDETIJMODE);
    return tdetid;

  } else {
    throw cms::Exception("InvalidDetId") << " Wrong indices in EcalElectronicsMapping::getTrigTowerDetId. TCCid = "
                                         << TCCid << " iTT = " << iTT << ".";
  }
}

EcalElectronicsId EcalElectronicsMapping::getElectronicsId(const DetId& id) const {
  EcalSubdetector subdet = EcalSubdetector(id.subdetId());
  if (subdet == EcalBarrel) {
    const EBDetId ebdetid = EBDetId(id);

    int dcc = DCCid(ebdetid);
    bool EBPlus = (zside(dcc, DCCMODE) > 0);
    bool EBMinus = !EBPlus;

    EcalTrigTowerDetId trigtower = ebdetid.tower();
    // int tower = trigtower.iTT();
    int tower = iTT(trigtower);

    int ieta = EBDetId(id).ietaAbs();
    int iphi = EBDetId(id).iphi();
    int strip(0);
    int channel(0);
    bool RightTower = rightTower(tower);
    if (RightTower) {
      strip = (ieta - 1) % 5;
      if (strip % 2 == 0) {
        if (EBMinus)
          channel = (iphi - 1) % 5;
        if (EBPlus)
          channel = 4 - ((iphi - 1) % 5);
      } else {
        if (EBMinus)
          channel = 4 - ((iphi - 1) % 5);
        if (EBPlus)
          channel = (iphi - 1) % 5;
      }
    } else {
      strip = 4 - ((ieta - 1) % 5);
      if (strip % 2 == 0) {
        if (EBMinus)
          channel = 4 - ((iphi - 1) % 5);
        if (EBPlus)
          channel = (iphi - 1) % 5;
      } else {
        if (EBMinus)
          channel = (iphi - 1) % 5;
        if (EBPlus)
          channel = 4 - ((iphi - 1) % 5);
      }
    }
    strip += 1;
    channel += 1;

    EcalElectronicsId elid = EcalElectronicsId(dcc, tower, strip, channel);

    return elid;
  } else if (subdet == EcalEndcap) {
    EcalElectronicsMap_by_DetId::const_iterator it = get<0>(m_items).find(id);
    if (it == get<0>(m_items).end()) {
      EcalElectronicsId elid(0);
      edm::LogError("EcalElectronicsMapping") << "Ecal mapping was asked non valid id";
      return elid;
    }
    EcalElectronicsId elid = it->elid;
    return elid;
  } else {
    throw cms::Exception("InvalidDetId") << " Wrong DetId in EcalElectronicsMapping::getElectronicsId.";
  }
}

EcalTriggerElectronicsId EcalElectronicsMapping::getTriggerElectronicsId(const DetId& id) const {
  EcalSubdetector subdet = EcalSubdetector(id.subdetId());

  if (subdet == EcalBarrel) {
    const EcalElectronicsId& elid = getElectronicsId(id);
    EcalTriggerElectronicsId trelid = getTriggerElectronicsId(elid);
    return trelid;
  } else if (subdet == EcalEndcap) {
    EcalElectronicsMap_by_DetId::const_iterator it = get<0>(m_items).find(id);
    if (it == get<0>(m_items).end()) {
      EcalTriggerElectronicsId trelid(0);
      edm::LogError("EcalElectronicsMapping") << "Ecal mapping was asked non valid trig id";
      return trelid;
    }
    EcalTriggerElectronicsId trelid = it->trelid;
    return trelid;
  } else {
    throw cms::Exception("InvalidDetId") << " Wrong DetId in EcalElectronicsMapping::getTriggerElectronicsId.";
  }
}

DetId EcalElectronicsMapping::getDetId(const EcalElectronicsId& id) const {
  EcalSubdetector subdet = id.subdet();

  if (subdet == EcalBarrel) {
    int dcc = id.dccId();
    int tower = id.towerId();
    int strip = id.stripId();
    int channel = id.xtalId();

    int smid = 0;
    int iphi = 0;
    bool EBPlus = (id.zside() > 0);
    bool EBMinus = !EBPlus;

    if (id.zside() < 0) {
      smid = dcc + 19 - DCCID_PHI0_EBM;
      iphi = (smid - 19) * kCrystalsInPhi;
      iphi += 5 * ((tower - 1) % kTowersInPhi);
    } else {
      smid = dcc + 1 - DCCID_PHI0_EBP;
      iphi = (smid - 1) * kCrystalsInPhi;
      iphi += 5 * (kTowersInPhi - ((tower - 1) % kTowersInPhi) - 1);
    }
    bool RightTower = rightTower(tower);
    int ieta = 5 * ((tower - 1) / kTowersInPhi) + 1;
    if (RightTower) {
      ieta += (strip - 1);
      if (strip % 2 == 1) {
        if (EBMinus)
          iphi += (channel - 1) + 1;
        if (EBPlus)
          iphi += (4 - (channel - 1)) + 1;
      } else {
        if (EBMinus)
          iphi += (4 - (channel - 1)) + 1;
        if (EBPlus)
          iphi += (channel - 1) + 1;
      }
    } else {
      ieta += 4 - (strip - 1);
      if (strip % 2 == 1) {
        if (EBMinus)
          iphi += (4 - (channel - 1)) + 1;
        if (EBPlus)
          iphi += (channel - 1) + 1;
      } else {
        if (EBMinus)
          iphi += (channel - 1) + 1;
        if (EBPlus)
          iphi += (4 - (channel - 1)) + 1;
      }
    }
    if (id.zside() < 0)
      ieta = -ieta;

    EBDetId e(ieta, iphi, EBDetId::ETAPHIMODE);
    return e;
  }

  else if (subdet == EcalEndcap) {
    EcalElectronicsMap_by_ElectronicsId::const_iterator it = get<1>(m_items).find(id);
    if (it == (get<1>(m_items).end())) {
      DetId cell(0);
      edm::LogError("EcalElectronicsMapping") << "Ecal mapping was asked non DetId";
      return cell;
    }
    DetId cell = it->cell;
    return cell;
  } else
    throw cms::Exception("InvalidDetId") << "Wrong EcalElectronicsId in EcalElectronicsMapping::getDetId.";
}

EcalTriggerElectronicsId EcalElectronicsMapping::getTriggerElectronicsId(const EcalElectronicsId& id) const {
  EcalSubdetector subdet = id.subdet();

  if (subdet == EcalBarrel) {
    int strip = id.stripId();
    int xtal = id.xtalId();
    int tower = id.towerId();
    int tcc = id.dccId();
    if (id.zside() < 0) {
      tcc += TCCID_PHI0_EBM - DCCID_PHI0_EBM;
    } else {
      tcc += TCCID_PHI0_EBP - DCCID_PHI0_EBP;
    }
    EcalTriggerElectronicsId trelid(tcc, tower, strip, xtal);
    return trelid;

  } else if (subdet == EcalEndcap) {
    EcalElectronicsMap_by_ElectronicsId::const_iterator it = get<1>(m_items).find(id);
    if (it == get<1>(m_items).end()) {
      EcalTriggerElectronicsId trelid(0);
      edm::LogError("EcalElectronicsMapping") << "Ecal mapping was asked non valid id";
      return trelid;
    }
    EcalTriggerElectronicsId trelid = it->trelid;
    return trelid;
  } else
    throw cms::Exception("InvalidDetId")
        << "Wrong EcalElectronicsId in EcalElectronicsMapping::getTriggerElectronicsId.";
}

DetId EcalElectronicsMapping::getDetId(const EcalTriggerElectronicsId& id) const {
  EcalSubdetector subdet = id.subdet();

  if (subdet == EcalBarrel) {
    const EcalElectronicsId& elid = getElectronicsId(id);
    DetId cell = getDetId(elid);
    return cell;
  } else if (subdet == EcalEndcap) {
    EcalElectronicsMap_by_TriggerElectronicsId::const_iterator it = get<2>(m_items).find(id);
    if (it == get<2>(m_items).end()) {
      DetId cell(0);
      edm::LogError("EcalElectronicsMapping") << "Ecal mapping was asked non valid DetId";
      return cell;
    }
    DetId cell = it->cell;
    return cell;
  } else
    throw cms::Exception("InvalidDetId") << "Wrong EcalTriggerElectronicsId in EcalElectronicsMapping::getDetId.";
}

EcalElectronicsId EcalElectronicsMapping::getElectronicsId(const EcalTriggerElectronicsId& id) const {
  EcalSubdetector subdet = id.subdet();

  if (subdet == EcalBarrel) {
    int strip = id.pseudoStripId();
    int xtal = id.channelId();
    int tower = id.ttId();
    int dcc = id.tccId();
    if (id.zside() < 0) {
      dcc -= TCCID_PHI0_EBM - DCCID_PHI0_EBM;
    } else {
      dcc -= TCCID_PHI0_EBP - DCCID_PHI0_EBP;
    }
    EcalElectronicsId elid(dcc, tower, strip, xtal);
    return elid;
  } else if (subdet == EcalEndcap) {
    EcalElectronicsMap_by_TriggerElectronicsId::const_iterator it = get<2>(m_items).find(id);
    if (it == get<2>(m_items).end()) {
      EcalElectronicsId elid(0);
      edm::LogError("EcalElectronicsMapping") << "Ecal mapping was asked non valid id";
      return elid;
    }
    EcalElectronicsId elid = it->elid;
    return elid;
  } else
    throw cms::Exception("InvalidDetId")
        << "Wrong EcalTriggerElectronicsId in EcalElectronicsMapping::getElectronicsId.";
}

std::vector<DetId> EcalElectronicsMapping::dccConstituents(int dccId) const {
  EcalSubdetector sub = subdet(dccId, DCCMODE);
  std::vector<DetId> items;

  if (sub == EcalBarrel) {
    for (int tower = 1; tower <= kEBTowersPerSM; tower++) {
      std::vector<DetId> xtals = dccTowerConstituents(dccId, tower);
      int size = xtals.size();
      for (int i = 0; i < size; i++) {
        DetId detid = xtals[i];
        items.emplace_back(detid);
      }
    }
    return items;
  } else if (sub == EcalEndcap) {
    EcalElectronicsMap_by_DccId::const_iterator lb, ub;
    boost::tuples::tie(lb, ub) = get<3>(m_items).equal_range(dccId);
    while (lb != ub) {
      DetId cell = lb->cell;
      items.emplace_back(cell);
      ++lb;
    }
    return items;
  } else
    throw cms::Exception("InvalidDetId") << "Wrong dccId = " << dccId
                                         << " in EcalElectronicsMapping::dccConstituents. ";
}

std::vector<DetId> EcalElectronicsMapping::dccTowerConstituents(int dccId, int tower) const {
  EcalSubdetector sub = subdet(dccId, DCCMODE);
  std::vector<DetId> items;

  if (sub == EcalBarrel) {
    int iz = zside(dccId, DCCMODE);
    int smid = 0;
    int iphi = 0;
    if (iz < 0) {
      smid = dccId + 19 - DCCID_PHI0_EBM;
      iphi = (smid - 19) * kCrystalsInPhi;
      iphi += 5 * ((tower - 1) % kTowersInPhi);
    } else {
      smid = dccId + 1 - DCCID_PHI0_EBP;
      iphi = (smid - 1) * kCrystalsInPhi;
      iphi += 5 * (kTowersInPhi - ((tower - 1) % kTowersInPhi) - 1);
    }
    int ieta = 5 * ((tower - 1) / kTowersInPhi) + 1;
    for (int ip = 1; ip <= 5; ip++) {
      for (int ie = 0; ie <= 4; ie++) {
        int ieta_xtal = ieta + ie;
        int iphi_xtal = iphi + ip;
        if (iz < 0)
          ieta_xtal = -ieta_xtal;
        EBDetId ebdetid(ieta_xtal, iphi_xtal, EBDetId::ETAPHIMODE);
        items.emplace_back(ebdetid);
      }
    }
    return items;
  }

  else if (sub == EcalEndcap) {
    EcalElectronicsMap_by_DccId_and_TowerId::const_iterator lb, ub;
    boost::tuples::tie(lb, ub) = get<4>(m_items).equal_range(boost::make_tuple(int(dccId), int(tower)));
    while (lb != ub) {
      DetId cell = lb->cell;
      items.emplace_back(cell);
      ++lb;
    }
    return items;
  } else
    throw cms::Exception("InvalidDetId") << "Wrong dccId = " << dccId << " tower = " << tower
                                         << " in EcalElectronicsMapping::dccTowerConstituents.";
}

std::vector<DetId> EcalElectronicsMapping::stripConstituents(int dccId, int tower, int strip) const {
  EcalSubdetector sub = subdet(dccId, DCCMODE);
  std::vector<DetId> items;

  if (sub == EcalBarrel) {
    int iz = zside(dccId, DCCMODE);
    bool RightTower = rightTower(tower);
    int smid = 0;
    int iphi = 0;
    if (iz < 0) {
      smid = dccId + 19 - DCCID_PHI0_EBM;
      iphi = (smid - 19) * kCrystalsInPhi;
      iphi += 5 * ((tower - 1) % kTowersInPhi);
    } else {
      smid = dccId + 1 - DCCID_PHI0_EBP;
      iphi = (smid - 1) * kCrystalsInPhi;
      iphi += 5 * (kTowersInPhi - ((tower - 1) % kTowersInPhi) - 1);
    }
    int ieta = 5 * ((tower - 1) / kTowersInPhi) + 1;
    if (RightTower) {
      ieta += (strip - 1);
    } else {
      ieta += 4 - (strip - 1);
    }
    for (int ip = 1; ip <= 5; ip++) {
      int ieta_xtal = ieta;
      int iphi_xtal = iphi + ip;
      if (iz < 0)
        ieta_xtal = -ieta_xtal;
      EBDetId ebdetid(ieta_xtal, iphi_xtal, EBDetId::ETAPHIMODE);
      items.emplace_back(ebdetid);
    }

    return items;
  } else {
    EcalElectronicsMap_by_DccId_TowerId_and_StripId::const_iterator lb, ub;
    boost::tuples::tie(lb, ub) = get<5>(m_items).equal_range(boost::make_tuple(int(dccId), int(tower), int(strip)));
    while (lb != ub) {
      DetId cell = lb->cell;
      items.emplace_back(cell);
      ++lb;
    }
    return items;
  }
}

std::vector<DetId> EcalElectronicsMapping::tccConstituents(int tccId) const {
  EcalSubdetector sub = subdet(tccId, TCCMODE);
  std::vector<DetId> items;

  if (sub == EcalBarrel) {
    int iz = zside(tccId, TCCMODE);
    int dccId = tccId;
    if (iz > 0)
      dccId = dccId - TCCID_PHI0_EBP + DCCID_PHI0_EBP;
    else
      dccId = dccId - TCCID_PHI0_EBM + DCCID_PHI0_EBM;
    items = dccConstituents(dccId);
    return items;
  } else {
    EcalElectronicsMap_by_TccId::const_iterator lb, ub;
    boost::tuples::tie(lb, ub) = get<6>(m_items).equal_range(tccId);
    while (lb != ub) {
      DetId cell = lb->cell;
      items.emplace_back(cell);
      ++lb;
    }
    return items;
  }
}

std::vector<DetId> EcalElectronicsMapping::ttConstituents(int tccId, int tt) const {
  EcalSubdetector sub = subdet(tccId, TCCMODE);
  std::vector<DetId> items;

  if (sub == EcalBarrel) {
    int iz = zside(tccId, TCCMODE);
    int dccId = tccId;
    if (iz > 0)
      dccId = dccId - TCCID_PHI0_EBP + DCCID_PHI0_EBP;
    else
      dccId = dccId - TCCID_PHI0_EBM + DCCID_PHI0_EBM;
    items = dccTowerConstituents(dccId, tt);
    return items;
  } else {
    EcalElectronicsMap_by_TccId_and_TtId::const_iterator lb, ub;
    boost::tuples::tie(lb, ub) = get<7>(m_items).equal_range(boost::make_tuple(int(tccId), int(tt)));
    while (lb != ub) {
      DetId cell = lb->cell;
      items.emplace_back(cell);
      ++lb;
    }
    return items;
  }
}

std::vector<DetId> EcalElectronicsMapping::pseudoStripConstituents(int tccId, int tt, int pseudostrip) const {
  EcalSubdetector sub = subdet(tccId, TCCMODE);
  std::vector<DetId> items;

  if (sub == EcalBarrel) {
    int iz = zside(tccId, TCCMODE);
    int dccId = tccId;
    if (iz > 0)
      dccId = dccId - TCCID_PHI0_EBP + DCCID_PHI0_EBP;
    else
      dccId = dccId - TCCID_PHI0_EBM + DCCID_PHI0_EBM;
    items = stripConstituents(dccId, tt, pseudostrip);
    return items;
  } else {
    EcalElectronicsMap_by_TccId_TtId_and_PseudostripId::const_iterator lb, ub;
    boost::tuples::tie(lb, ub) = get<8>(m_items).equal_range(boost::make_tuple(int(tccId), int(tt), int(pseudostrip)));
    while (lb != ub) {
      DetId cell = lb->cell;
      items.emplace_back(cell);
      ++lb;
    }
    return items;
  }
}

void EcalElectronicsMapping::assign(const DetId& cell,
                                    const EcalElectronicsId& elid,
                                    const EcalTriggerElectronicsId& tower) {
  m_items.insert(MapItem(cell, elid, tower));
}

std::pair<int, int> EcalElectronicsMapping::getDCCandSC(EcalScDetId id) const {
  // pair.first = DCC id
  // pair.second = DCC_channel
  // For digi2raw, read the SRflags and write the SR block :
  // need to find out, for an EcalScDetId, which is the DCC and the DCC_channel

  std::pair<int, int> ind;
  EEDetId dum;
  int ix = id.ix();
  int iy = id.iy();
  int zside = id.zside();
  ix = (ix - 1) * 5 + 1;
  iy = (iy - 1) * 5 + 1;
  ix = 5 * (ix / 5) + 1;
  iy = 5 * (iy / 5) + 1;
  int ix_c = ix;
  int iy_c = iy;
  if (!dum.validDetId(ix_c, iy_c, zside)) {
    ix_c = ix + 4;
    iy_c = iy;
    if (!dum.validDetId(ix_c, iy_c, zside)) {
      ix_c = ix + 4;
      iy_c = iy + 4;
      if (!dum.validDetId(ix_c, iy_c, zside)) {
        ix_c = ix;
        iy_c = iy + 4;
      }
    }
  }
  EEDetId eedetid(ix_c, iy_c, zside, EEDetId::XYMODE);
  EcalElectronicsId elid = getElectronicsId(eedetid);
  int Dccid = elid.dccId();
  int DCC_Channel = elid.towerId();
  ind.first = Dccid;
  ind.second = DCC_Channel;
  return ind;
}

std::vector<EcalScDetId> EcalElectronicsMapping::getEcalScDetId(int DCCid,
                                                                int DCC_Channel,
                                                                bool ignoreSingleCrystal) const {
  //Debug output switch:
  const bool debug = false;

  // For unpacking of ST flags.
  //result: SCs readout by the DCC channel DCC_channel of DCC DCCid.
  //Vector of 1 or 2 elements: most of the time there is only
  //one SC read-out by the DCC channel, but for some channels
  //there are 2 partial SCs which were grouped.
  std::vector<EcalScDetId> scDetIds;

  //There are 4 SCs in each endcap whose one crystal is read out
  //by a different DCC channel than the others.
  //Number of crystals of the SC read out by the DCC channel:
  std::vector<int> nReadoutXtals;

  std::vector<DetId> xtals = dccTowerConstituents(DCCid, DCC_Channel);

  if (debug) {
    std::ostringstream st1;
    st1 << __FILE__ << ":" << __LINE__ << ": " << xtals.size() << " crystals read out by channel " << DCC_Channel << " of DCC " << DCCid << ": ";
    for (auto xtal : xtals) {
      st1 << EEDetId(xtal) << " ";
    }
    edm::LogVerbatim("EcalMapping") << st1.str() << "\n";
  }

  if (xtals.empty())
    throw cms::Exception("InvalidDetId") << "EcalElectronicsMapping : can not create EcalScDetId for DCC " << DCCid
                                         << " and DCC_Channel " << DCC_Channel << ".";

  for (auto xtal : xtals) {
    EEDetId eedetid = xtal;
    int ix = eedetid.ix();
    int iy = eedetid.iy();
    int iz = eedetid.zside();
    int ix_SC = (ix - 1) / 5 + 1;
    int iy_SC = (iy - 1) / 5 + 1;
    //Supercrystal (SC) the crystal belongs to:
    EcalScDetId scdetid(ix_SC, iy_SC, iz);
    size_t iSc = 0;
    //look if the SC was already included:
    while (iSc < scDetIds.size() && scDetIds[iSc] != scdetid)
      ++iSc;
    if (iSc == scDetIds.size()) {  //SC not yet included
      scDetIds.emplace_back(scdetid);
      nReadoutXtals.emplace_back(1);  //crystal counter of the added SC
    } else {                          //SC already included
      ++nReadoutXtals[iSc];           // counting crystals in the SC
    }
  }

  if (ignoreSingleCrystal) {
    //For simplification, SC read out by two different DCC channels
    //will be associated to the DCC channel reading most of the crystals,
    //the other DCC channel which read only one crystal is discarded.
    //Discards SC with only one crystal read out by the considered,
    //DCC channel:
    assert(scDetIds.size() == nReadoutXtals.size());
    for (size_t iSc = 0; iSc < scDetIds.size(); /*NOOP*/) {
      if (nReadoutXtals[iSc] <= 1) {
        if (debug)
          edm::LogVerbatim("EcalMapping") << "EcalElectronicsMapping::getEcalScDetId: Ignore SC " << scDetIds[iSc]
                    << " whose only one channel is read out by "
                       "the DCC channel (DCC "
                    << DCCid << ", ch " << DCC_Channel << ").\n";
        scDetIds.erase(scDetIds.begin() + iSc);
        nReadoutXtals.erase(nReadoutXtals.begin() + iSc);
      } else {
        ++iSc;  //next SC;
      }
    }
  }

  return scDetIds;
}

EcalSubdetector EcalElectronicsMapping::subdet(int dcctcc, int mode) const {
  if (mode == DCCMODE) {
    if ((dcctcc >= MIN_DCCID_EBM && dcctcc <= MAX_DCCID_EBM) || (dcctcc >= MIN_DCCID_EBP && dcctcc <= MAX_DCCID_EBP))
      return EcalBarrel;
    else
      return EcalEndcap;
  } else if (mode == TCCMODE) {
    if ((dcctcc >= MIN_TCCID_EBM && dcctcc <= MAX_TCCID_EBM) || (dcctcc >= MIN_TCCID_EBP && dcctcc <= MAX_TCCID_EBP))
      return EcalBarrel;
    else
      return EcalEndcap;
  } else
    throw cms::Exception("InvalidDetId") << " Wrong mode in EcalElectronicsMapping::subdet " << mode << ".";
}

int EcalElectronicsMapping::zside(int dcctcc, int mode) const {
  if (mode == DCCMODE) {
    if (dcctcc >= MIN_DCCID_EBM && dcctcc <= MAX_DCCID_EBM)
      return -1;
    if (dcctcc >= MIN_DCCID_EBP && dcctcc <= MAX_DCCID_EBP)
      return +1;
    if (dcctcc >= MIN_DCCID_EEM && dcctcc <= MAX_DCCID_EEM)
      return -1;
    if (dcctcc >= MIN_DCCID_EEP && dcctcc <= MAX_DCCID_EEP)
      return +1;
  } else if (mode == TCCMODE) {
    if (dcctcc >= MIN_TCCID_EBM && dcctcc <= MAX_TCCID_EBM)
      return -1;
    if (dcctcc >= MIN_TCCID_EBP && dcctcc <= MAX_TCCID_EBP)
      return +1;
    if (dcctcc >= MIN_TCCID_EEM && dcctcc <= MAX_TCCID_EEM)
      return -1;
    if (dcctcc >= MIN_TCCID_EEP && dcctcc <= MAX_TCCID_EEP)
      return +1;
  } else {
    throw cms::Exception("InvalidDetId") << " Wrong mode in EcalElectronicsMapping::zside " << mode << ".";
  }
  return 0;
}

bool EcalElectronicsMapping::rightTower(int tower) const {
  // for EB, two types of tower (LVRB top/bottom)

  if ((tower > 12 && tower < 21) || (tower > 28 && tower < 37) || (tower > 44 && tower < 53) ||
      (tower > 60 && tower < 69))
    return true;
  else
    return false;
}

int EcalElectronicsMapping::DCCBoundary(int FED) const {
  if (FED >= MIN_DCCID_EEM && FED <= MAX_DCCID_EEM)
    return MIN_DCCID_EEM;
  if (FED >= MIN_DCCID_EBM && FED <= MAX_DCCID_EBM)
    return MIN_DCCID_EBM;
  if (FED >= MIN_DCCID_EBP && FED <= MAX_DCCID_EBP)
    return MIN_DCCID_EBP;
  if (FED >= MIN_DCCID_EEP && FED <= MAX_DCCID_EEP)
    return MIN_DCCID_EEP;
  return -1;
}

std::vector<int> EcalElectronicsMapping::GetListofFEDs(const RectangularEtaPhiRegion& region) const {
  std::vector<int> FEDs;
  GetListofFEDs(region, FEDs);
  return FEDs;
}
void EcalElectronicsMapping::GetListofFEDs(const RectangularEtaPhiRegion& region, std::vector<int>& FEDs) const {
  // for regional unpacking.
  // get list of FEDs corresponding to a region in (eta,phi)

  //	std::vector<int> FEDs;
  double radTodeg = 180. / M_PI;
  ;

  bool debug = false;

  double etalow = region.etaLow();
  double philow = region.phiLow() * radTodeg;
  if (debug)
    edm::LogVerbatim("EcalMapping") << " etalow philow " << etalow << " " << philow;
  int FED_LB = GetFED(etalow, philow);  // left, bottom

  double phihigh = region.phiHigh() * radTodeg;
  if (debug)
    edm::LogVerbatim("EcalMapping") << " etalow phihigh " << etalow << " " << phihigh;
  int FED_LT = GetFED(etalow, phihigh);  // left, top

  int DCC_BoundaryL = DCCBoundary(FED_LB);
  int deltaL = 18;
  if (FED_LB < MIN_DCCID_EBM || FED_LB > MAX_DCCID_EBP)
    deltaL = 9;

  if (philow < -170 && phihigh > 170) {
    FED_LB = DCC_BoundaryL;
    FED_LT = DCC_BoundaryL + deltaL - 1;
  }
  if (debug)
    edm::LogVerbatim("EcalMapping") << " FED_LB FED_LT " << FED_LB << " " << FED_LT;

  bool dummy = true;
  int idx = 0;
  while (dummy) {
    int iL = (FED_LB - DCC_BoundaryL + idx) % deltaL + DCC_BoundaryL;
    FEDs.emplace_back(iL);
    if (debug)
      edm::LogVerbatim("EcalMapping") << "   add fed " << iL;
    if (iL == FED_LT)
      break;
    idx++;
  }

  double etahigh = region.etaHigh();
  int FED_RB = GetFED(etahigh, philow);  // right, bottom
  if (FED_RB == FED_LB)
    return;  // FEDs;

  int FED_RT = GetFED(etahigh, phihigh);  // right, top

  if (debug)
    edm::LogVerbatim("EcalMapping") << "etahigh philow phihigh " << etahigh << " " << philow << " " << phihigh;
  int DCC_BoundaryR = DCCBoundary(FED_RB);
  int deltaR = 18;
  if (FED_RB < MIN_DCCID_EBM || FED_RB > MAX_DCCID_EBP)
    deltaR = 9;

  if (philow < -170 && phihigh > 170) {
    FED_RB = DCC_BoundaryR;
    FED_RT = DCC_BoundaryR + deltaR - 1;
  }
  if (debug)
    edm::LogVerbatim("EcalMapping") << " FED_RB FED_RT " << FED_RB << " " << FED_RT;
  idx = 0;
  while (dummy) {
    int iR = (FED_RB - DCC_BoundaryR + idx) % deltaR + DCC_BoundaryR;
    FEDs.emplace_back(iR);
    if (debug)
      edm::LogVerbatim("EcalMapping") << "   add fed " << iR;
    if (iR == FED_RT)
      break;
    idx++;
  }

  if (FED_LB >= MIN_DCCID_EBM && FED_LB <= MAX_DCCID_EBM && FED_RB >= MIN_DCCID_EEP && FED_RB <= MAX_DCCID_EEP) {
    int minR = FED_LB + 18;
    int maxR = FED_LT + 18;
    int idx = 0;
    while (dummy) {
      int iR = (minR - MIN_DCCID_EBP + idx) % 18 + MIN_DCCID_EBP;
      FEDs.emplace_back(iR);
      if (debug)
        edm::LogVerbatim("EcalMapping") << "   add fed " << iR;
      if (iR == maxR)
        break;
      idx++;
    }
    return;  // FEDs;
  }

  if (FED_LB >= MIN_DCCID_EEM && FED_LB <= MAX_DCCID_EEM && FED_RB >= MIN_DCCID_EBP && FED_RB <= MAX_DCCID_EBP) {
    int minL = FED_RB - 18;
    int maxL = FED_RT - 18;
    int idx = 0;
    while (dummy) {
      int iL = (minL - MIN_DCCID_EBM + idx) % 18 + MIN_DCCID_EBM;
      FEDs.emplace_back(iL);
      if (debug)
        edm::LogVerbatim("EcalMapping") << "   add fed " << iL;
      if (iL == maxL)
        break;
      idx++;
    }
    return;  // FEDs;
  }

  if (FED_LB >= MIN_DCCID_EEM && FED_LB <= MAX_DCCID_EEM && FED_RB >= MIN_DCCID_EEP && FED_RB <= MAX_DCCID_EEP) {
    int minL = (FED_LB - 1) * 2 + MIN_DCCID_EBM;
    if (minL == MIN_DCCID_EBM)
      minL = MAX_DCCID_EBM;
    else
      minL = minL - 1;
    int maxL = (FED_LT - 1) * 2 + MIN_DCCID_EBM;
    int idx = 0;
    while (dummy) {
      int iL = (minL - MIN_DCCID_EBM + idx) % 18 + MIN_DCCID_EBM;
      FEDs.emplace_back(iL);
      if (debug)
        edm::LogVerbatim("EcalMapping") << "   add fed " << iL;
      if (iL == maxL)
        break;
      idx++;
    }
    int minR = minL + 18;
    int maxR = maxL + 18;
    idx = 0;
    while (dummy) {
      int iR = (minR - MIN_DCCID_EBP + idx) % 18 + MIN_DCCID_EBP;
      FEDs.emplace_back(iR);
      if (debug)
        edm::LogVerbatim("EcalMapping") << "   add fed " << iR;
      if (iR == maxR)
        break;
      idx++;
    }
  }

  return;  // FEDs;
}

int EcalElectronicsMapping::GetFED(double eta, double phi) const {
  // for regional unpacking.
  // eta is signed, phi is in degrees.

  int DCC_Phi0 = 0;
  bool IsBarrel = true;
  if (fabs(eta) > 1.479)
    IsBarrel = false;
  bool Positive = (eta > 0);

  if (IsBarrel && Positive)
    DCC_Phi0 = DCCID_PHI0_EBP;
  if (IsBarrel && (!Positive))
    DCC_Phi0 = DCCID_PHI0_EBM;
  if ((!IsBarrel) && Positive)
    DCC_Phi0 = MIN_DCCID_EEP;
  if ((!IsBarrel) && (!Positive))
    DCC_Phi0 = MIN_DCCID_EEM;

  // phi between 0 and 360 deg :
  if (phi < 0)
    phi += 360;
  if (phi > 360.)
    phi = 360.;
  if (phi < 0)
    phi = 0.;

  if (IsBarrel)
    phi = phi - 350;
  else
    phi = phi - 330;
  if (phi < 0)
    phi += 360;
  int iphi = -1;
  if (IsBarrel)
    iphi = (int)(phi / 20.);
  else
    iphi = (int)(phi / 40.);

  // edm::LogVerbatim("EcalMapping") << " in GetFED : phi iphi DCC0 " << phi << " " << iphi << " " << DCC_Phi0;

  int DCC = iphi + DCC_Phi0;
  // edm::LogVerbatim("EcalMapping") << "  eta phi " << eta << " " << " " << phi << " is in FED " << DCC;
  return DCC;
}

int EcalElectronicsMapping::getLMNumber(const DetId& id) const {
  // Laser Monitoring readout number.

  EcalSubdetector subdet = EcalSubdetector(id.subdetId());

  if (subdet == EcalBarrel) {
    const EBDetId ebdetid = EBDetId(id);
    int dccid = DCCid(ebdetid);
    std::map<int, int>::const_iterator it = LaserMonitoringMap_EB.find(dccid);
    if (it != LaserMonitoringMap_EB.end()) {
      int ilm = it->second;
      int iETA = ebdetid.ietaSM();
      int iPHI = ebdetid.iphiSM();
      if (iPHI > 10 && iETA > 5) {
        ilm++;
      };
      return ilm;
    } else
      throw cms::Exception("InvalidDCCId") << "Wrong DCCId (EB) in EcalElectronicsMapping::getLMNumber.";
  }

  else if (subdet == EcalEndcap) {
    EcalElectronicsId elid = getElectronicsId(id);
    int dccid = elid.dccId();
    EEDetId eedetid = EEDetId(id);
    std::map<int, int>::const_iterator it = LaserMonitoringMap_EE.find(dccid);
    if (it != LaserMonitoringMap_EB.end()) {
      int ilm = it->second;
      if (dccid == 8) {
        int ix = eedetid.ix();
        if (ix > 50)
          ilm += 1;
      }
      if (dccid == 53) {
        int ix = eedetid.ix();
        if (ix > 50)
          ilm += 1;
      }
      return ilm;
    } else
      throw cms::Exception("InvalidDCCId") << "Wrong DCCId (EE) in EcalElectronicsMapping::getLMNumber.";
  }

  return -1;
}
