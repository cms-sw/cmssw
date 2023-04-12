#include "Geometry/ForwardGeometry/interface/CastorTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>
#include <iostream>
#include <algorithm>

static const int MODULE_EM_MAX = 2;
static const int MODULE_HAD_MAX = 12;

CastorTopology::CastorTopology()
    : excludeEM_(false),
      excludeHAD_(false),
      excludeZP_(false),
      excludeZN_(false),
      firstEMModule_(1),
      lastEMModule_(2),
      firstHADModule_(3),
      lastHADModule_(14) {}

bool CastorTopology::valid(const HcalCastorDetId& id) const { return (validRaw(id) && !isExcluded(id)); }

bool CastorTopology::isExcluded(const HcalCastorDetId& id) const {
  bool exed = false;

  // check for section exclutions
  switch (id.section()) {
    case (HcalCastorDetId::EM):
      exed = excludeEM_;
      break;
    case (HcalCastorDetId::HAD):
      exed = excludeHAD_;
      break;
    default:
      exed = false;
  }

  // check the entire list
  if (!exed && !exclusionList_.empty()) {
    std::vector<HcalCastorDetId>::const_iterator i = std::lower_bound(exclusionList_.begin(), exclusionList_.end(), id);
    if (i != exclusionList_.end() && *i == id)
      exed = true;
  }
  return exed;
}

void CastorTopology::exclude(const HcalCastorDetId& id) {
  std::vector<HcalCastorDetId>::iterator i = std::lower_bound(exclusionList_.begin(), exclusionList_.end(), id);
  if (i == exclusionList_.end() || *i != id) {
    exclusionList_.insert(i, id);
  }
}

void CastorTopology::exclude(int zside) {
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

void CastorTopology::exclude(int zside, HcalCastorDetId::Section section) {
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
    case (HcalCastorDetId::EM):
      excludeEM_ = true;
      break;
    case (HcalCastorDetId::HAD):
      excludeHAD_ = true;
      break;
    default:
      break;
  }
}

int CastorTopology::exclude(int zside,
                            HcalCastorDetId::Section section1,
                            int isec1,
                            int imod1,
                            HcalCastorDetId::Section section2,
                            int isec2,
                            int imod2) {
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

  /* NOTE not so sure about the exclusion */
  if (section1 == HcalCastorDetId::EM && section2 == HcalCastorDetId::EM) {
    exed = excludeEM_;
  } else if (section1 == HcalCastorDetId::HAD && section2 == HcalCastorDetId::HAD) {
    exed = excludeHAD_;
  } else {
    exed = false;
  };

  if (exed)
    return 0;

  bool isPositive = false;
  if (zside == 1)
    isPositive = true;

  int n = 0;
  for (int isec = isec1; isec < isec2; isec++) {
    for (int imod = imod1; imod < imod2; imod++) {
      HcalCastorDetId id(section1, isPositive, isec, imod);
      if (validRaw(id))
        exclude(id);
      n++;
    }
  }
  return n;
}

bool CastorTopology::validRaw(const HcalCastorDetId& id) const {
  return HcalCastorDetId::validDetId(id.section(), id.zside() > 0, id.sector(), id.module());
}

std::vector<DetId> CastorTopology::incSector(const DetId& id) const {
  std::vector<DetId> vNeighborsDetId;
  HcalCastorDetId castorId = HcalCastorDetId(id);
  HcalCastorDetId castorDetId;
  if (validRaw(castorId)) {
    bool isPositive = false;
    if (castorId.zside() == 1)
      isPositive = true;
    if (castorId.sector() == 1) {
      castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector() + 1, castorId.module());
      vNeighborsDetId.emplace_back(castorDetId.rawId());
      return vNeighborsDetId;
    }
    if (castorId.sector() == 16) {
      castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector() - 1, castorId.module());
      vNeighborsDetId.emplace_back(castorDetId.rawId());
      return vNeighborsDetId;
    }
    castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector() - 1, castorId.module());
    vNeighborsDetId.emplace_back(castorDetId.rawId());
    castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector() + 1, castorId.module());
    vNeighborsDetId.emplace_back(castorDetId.rawId());
  }
  return vNeighborsDetId;
}

std::vector<DetId> CastorTopology::incModule(const DetId& id) const {
  std::vector<DetId> vNeighborsDetId;
  HcalCastorDetId castorId = HcalCastorDetId(id);
  HcalCastorDetId castorDetId;
  if (validRaw(castorId) && castorId.section() == HcalCastorDetId::EM) {
    bool isPositive = false;
    if (castorId.zside() == 1)
      isPositive = true;
    if (castorId.module() == 1) {
      castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector(), castorId.module() + 1);
      vNeighborsDetId.emplace_back(castorDetId.rawId());
      return vNeighborsDetId;
    }
    if (castorId.module() == MODULE_EM_MAX) {
      castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector(), castorId.module() - 1);
      vNeighborsDetId.emplace_back(castorDetId.rawId());
      return vNeighborsDetId;
    }
    castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector(), castorId.module() - 1);
    vNeighborsDetId.emplace_back(castorDetId.rawId());
    castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector(), castorId.module() + 1);
    vNeighborsDetId.emplace_back(castorDetId.rawId());
  }
  if (validRaw(castorId) && castorId.section() == HcalCastorDetId::HAD) {
    bool isPositive = false;
    if (castorId.zside() == 1)
      isPositive = true;
    if (castorId.module() == 1) {
      castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector(), castorId.module() + 1);
      vNeighborsDetId.emplace_back(castorDetId.rawId());
      return vNeighborsDetId;
    }
    if (castorId.module() == MODULE_HAD_MAX) {
      castorDetId = HcalCastorDetId(castorId.section(), isPositive, castorId.sector(), castorId.module() - 1);
      vNeighborsDetId.emplace_back(castorDetId.rawId());
      return vNeighborsDetId;
    }
  }
  return vNeighborsDetId;
}

std::vector<DetId> CastorTopology::east(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "CastorTopology::east() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}

std::vector<DetId> CastorTopology::west(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "CastorTopology::west() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}

std::vector<DetId> CastorTopology::north(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "CastorTopology::north() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}
std::vector<DetId> CastorTopology::south(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "CastorTopology::south() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}
std::vector<DetId> CastorTopology::up(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "CastorTopology::up() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}
std::vector<DetId> CastorTopology::down(const DetId& /*id*/) const {
  edm::LogVerbatim("ForwardGeom") << "CastorTopology::down() not yet implemented";
  std::vector<DetId> vNeighborsDetId;
  return vNeighborsDetId;
}

int CastorTopology::ncells(HcalCastorDetId::Section section) const {
  int ncells = 0;
  switch (section) {
    case (HcalCastorDetId::EM):
      ncells = MODULE_EM_MAX * 16;
      break;
    case (HcalCastorDetId::HAD):
      ncells = MODULE_HAD_MAX * 16;
      break;
    case (HcalCastorDetId::Unknown):
      ncells = 0;
      break;
  }
  return ncells;
}

int CastorTopology::firstCell(HcalCastorDetId::Section section) const {
  int firstCell = 0;
  switch (section) {
    case (HcalCastorDetId::EM):
      firstCell = firstEMModule_;
      break;
    case (HcalCastorDetId::HAD):
      firstCell = firstHADModule_;
      break;
    case (HcalCastorDetId::Unknown):
      firstCell = 0;
      break;
  }
  return firstCell;
}

int CastorTopology::lastCell(HcalCastorDetId::Section section) const {
  int lastCell = 0;
  switch (section) {
    case (HcalCastorDetId::EM):
      lastCell = lastEMModule_;
      break;
    case (HcalCastorDetId::HAD):
      lastCell = lastHADModule_;
      break;
    case (HcalCastorDetId::Unknown):
      lastCell = 0;
      break;
  }
  return lastCell;
}
