#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>
#include <iostream>
#include <algorithm>

static const int ICH_EM_MAX = 5;
static const int ICH_HAD_MAX = 4;
static const int ICH_LUM_MAX = 2;
static const int ICH_RPD_MAX = 16;

ZdcTopology::ZdcTopology()
    : excludeEM_(false),
      excludeHAD_(false),
      excludeLUM_(false),
      excludeRPD_(false),
      excludeZP_(false),
      excludeZN_(false),
      firstEMModule_(1),
      lastEMModule_(5),
      firstHADModule_(1),
      lastHADModule_(4),
      firstLUMModule_(1),
      lastLUMModule_(2),
      firstRPDModule_(1),
      lastRPDModule_(16) {}

bool ZdcTopology::valid(const HcalZDCDetId& id) const {
  // check the raw rules
  bool ok = validRaw(id);
  ok = ok && !isExcluded(id);
  return ok;
}

bool ZdcTopology::isExcluded(const HcalZDCDetId& id) const {
  bool exed = false;

  // check for section exclutions
  switch (id.section()) {
    case (HcalZDCDetId::EM):
      exed = excludeEM_;
      break;
    case (HcalZDCDetId::HAD):
      exed = excludeHAD_;
      break;
    case (HcalZDCDetId::LUM):
      exed = excludeLUM_;
      break;
    case (HcalZDCDetId::RPD):
      exed = excludeRPD_;
      break;
    default:
      exed = false;
  }

  // check the entire list
  if (!exed && !exclusionList_.empty()) {
    std::vector<HcalZDCDetId>::const_iterator i = std::lower_bound(exclusionList_.begin(), exclusionList_.end(), id);
    if (i != exclusionList_.end() && *i == id)
      exed = true;
  }
  return exed;
}

void ZdcTopology::exclude(const HcalZDCDetId& id) {
  std::vector<HcalZDCDetId>::iterator i = std::lower_bound(exclusionList_.begin(), exclusionList_.end(), id);
  if (i == exclusionList_.end() || *i != id) {
    exclusionList_.insert(i, id);
  }
}

void ZdcTopology::exclude(int zside) {
  switch (zside) {
    case (1):
      excludeZP_ = true;
      break;
    case (-1):
      excludeZN_ = true;
      break;
    default:
      break;
  }
}

void ZdcTopology::exclude(int zside, HcalZDCDetId::Section section) {
  switch (zside) {
    case (1):
      excludeZP_ = true;
      break;
    case (-1):
      excludeZN_ = true;
      break;
    default:
      break;
  }
  switch (section) {
    case (HcalZDCDetId::EM):
      excludeEM_ = true;
      break;
    case (HcalZDCDetId::HAD):
      excludeHAD_ = true;
      break;
    case (HcalZDCDetId::LUM):
      excludeLUM_ = true;
      break;
    case (HcalZDCDetId::RPD):
      excludeRPD_ = true;
      break;
    default:
      break;
  }
}

int ZdcTopology::exclude(int zside, HcalZDCDetId::Section section, int ich1, int ich2) {
  bool exed = false;
  switch (zside) {
    case (1):
      exed = excludeZP_;
      break;
    case (-1):
      exed = excludeZN_;
      break;
    default:
      exed = false;
  }
  if (exed)
    return 0;

  switch (section) {
    case (HcalZDCDetId::EM):
      exed = excludeEM_;
      break;
    case (HcalZDCDetId::HAD):
      exed = excludeHAD_;
      break;
    case (HcalZDCDetId::LUM):
      exed = excludeLUM_;
      break;
    case (HcalZDCDetId::RPD):
      exed = excludeRPD_;
      break;
    default:
      exed = false;
  }
  if (exed)
    return 0;

  bool isPositive = false;
  if (zside == 1)
    isPositive = true;

  int n = 0;
  for (int ich = ich1; ich < ich2; ich++) {
    HcalZDCDetId id(section, isPositive, ich);
    if (validRaw(id))
      exclude(id);
    n++;
  }
  return n;
}

bool ZdcTopology::validRaw(const HcalZDCDetId& id) const {
  bool ok = true;
  if (abs(id.zside()) != 1)
    ok = false;
  else if (id.channel() <= 0)
    ok = false;
  else if (!(id.section() == HcalZDCDetId::EM || id.section() == HcalZDCDetId::HAD ||
             id.section() == HcalZDCDetId::LUM))
    //	    id.section()== HcalZDCDetId::LUM ||
    //	    id.section()== HcalZDCDetId::RPD))
    ok = false;
  else if (id.section() == HcalZDCDetId::EM && id.channel() > ICH_EM_MAX)
    ok = false;
  else if (id.section() == HcalZDCDetId::HAD && id.channel() > ICH_HAD_MAX)
    ok = false;
  else if (id.section() == HcalZDCDetId::LUM && id.channel() > ICH_LUM_MAX)
    ok = false;
  else if (id.section() == HcalZDCDetId::RPD && id.channel() > ICH_RPD_MAX)
    ok = false;
  return ok;
}

std::vector<DetId> ZdcTopology::transverse(const DetId& id) const {
  std::vector<DetId> vNeighborsDetId;
  HcalZDCDetId zdcId = HcalZDCDetId(id);
  HcalZDCDetId zdcDetId;
  if (validRaw(zdcId) && zdcId.section() == HcalZDCDetId::EM) {
    bool isPositive = false;
    if (zdcId.zside() == 1)
      isPositive = true;
    if (zdcId.channel() == 1) {
      zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() + 1);
      vNeighborsDetId.emplace_back(zdcDetId.rawId());
      return vNeighborsDetId;
    }
    if (zdcId.channel() == ICH_EM_MAX) {
      zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() - 1);
      vNeighborsDetId.emplace_back(zdcDetId.rawId());
      return vNeighborsDetId;
    }
    zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() - 1);
    vNeighborsDetId.emplace_back(zdcDetId.rawId());
    zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() + 1);
    vNeighborsDetId.emplace_back(zdcDetId.rawId());
  }
  return vNeighborsDetId;
}

std::vector<DetId> ZdcTopology::longitudinal(const DetId& id) const {
  std::vector<DetId> vNeighborsDetId;
  HcalZDCDetId zdcId = HcalZDCDetId(id);
  HcalZDCDetId zdcDetId;
  if (validRaw(zdcId) && zdcId.section() == HcalZDCDetId::HAD) {
    bool isPositive = false;
    if (zdcId.zside() == 1)
      isPositive = true;
    if (zdcId.channel() == 1) {
      zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() + 1);
      vNeighborsDetId.emplace_back(zdcDetId.rawId());
      return vNeighborsDetId;
    }
    if (zdcId.channel() == ICH_HAD_MAX) {
      zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() - 1);
      vNeighborsDetId.emplace_back(zdcDetId.rawId());
      return vNeighborsDetId;
    }
    zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() - 1);
    vNeighborsDetId.emplace_back(zdcDetId.rawId());
    zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() + 1);
    vNeighborsDetId.emplace_back(zdcDetId.rawId());
  }
  if (validRaw(zdcId) && zdcId.section() == HcalZDCDetId::LUM) {
    bool isPositive = false;
    if (zdcId.zside() == 1)
      isPositive = true;
    if (zdcId.channel() == 1) {
      zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() + 1);
      vNeighborsDetId.emplace_back(zdcDetId.rawId());
      return vNeighborsDetId;
    }
    if (zdcId.channel() == ICH_LUM_MAX) {
      zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() - 1);
      vNeighborsDetId.emplace_back(zdcDetId.rawId());
      return vNeighborsDetId;
    }
  }
  if (validRaw(zdcId) && zdcId.section() == HcalZDCDetId::RPD) {
    bool isPositive = false;
    if (zdcId.zside() == 1)
      isPositive = true;
    if (zdcId.channel() == 1) {
      zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() + 1);
      vNeighborsDetId.emplace_back(zdcDetId.rawId());
      return vNeighborsDetId;
    }
    if (zdcId.channel() == ICH_RPD_MAX) {
      zdcDetId = HcalZDCDetId(zdcId.section(), isPositive, zdcId.channel() - 1);
      vNeighborsDetId.emplace_back(zdcDetId.rawId());
      return vNeighborsDetId;
    }
  }
  return vNeighborsDetId;
}

std::vector<DetId> ZdcTopology::east(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "ZdcTopology::east() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}

std::vector<DetId> ZdcTopology::west(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "ZdcTopology::west() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}

std::vector<DetId> ZdcTopology::north(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "ZdcTopology::north() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}
std::vector<DetId> ZdcTopology::south(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "ZdcTopology::south() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}
std::vector<DetId> ZdcTopology::up(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "ZdcTopology::up() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}
std::vector<DetId> ZdcTopology::down(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "ZdcTopology::down() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}

int ZdcTopology::ncells(HcalZDCDetId::Section section) const {
  int ncells = 0;
  switch (section) {
    case (HcalZDCDetId::EM):
      ncells = ICH_EM_MAX;
      break;
    case (HcalZDCDetId::HAD):
      ncells = ICH_HAD_MAX;
      break;
    case (HcalZDCDetId::LUM):
      ncells = ICH_LUM_MAX;
      break;
    case (HcalZDCDetId::RPD):
      ncells = ICH_RPD_MAX;
      break;
    case (HcalZDCDetId::Unknown):
      ncells = 0;
      break;
  }
  return ncells;
}

int ZdcTopology::firstCell(HcalZDCDetId::Section section) const {
  int firstCell = 0;
  switch (section) {
    case (HcalZDCDetId::EM):
      firstCell = firstEMModule_;
      break;
    case (HcalZDCDetId::HAD):
      firstCell = firstHADModule_;
      break;
    case (HcalZDCDetId::LUM):
      firstCell = firstLUMModule_;
      break;
    case (HcalZDCDetId::RPD):
      firstCell = firstRPDModule_;
      break;
    case (HcalZDCDetId::Unknown):
      firstCell = 0;
      break;
  }
  return firstCell;
}

int ZdcTopology::lastCell(HcalZDCDetId::Section section) const {
  int lastCell = 0;
  switch (section) {
    case (HcalZDCDetId::EM):
      lastCell = lastEMModule_;
      break;
    case (HcalZDCDetId::HAD):
      lastCell = lastHADModule_;
      break;
    case (HcalZDCDetId::LUM):
      lastCell = lastLUMModule_;
      break;
    case (HcalZDCDetId::RPD):
      lastCell = lastRPDModule_;
      break;
    case (HcalZDCDetId::Unknown):
      lastCell = 0;
      break;
  }
  return lastCell;
}
