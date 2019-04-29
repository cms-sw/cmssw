#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"


CaloTopology::CaloTopology() {
}

CaloTopology::~CaloTopology() = default;


int CaloTopology::makeIndex(DetId::Detector det, int subdet) const {
  return (int(det)<<4) | (subdet&0xF);
}

void CaloTopology::setSubdetTopology(DetId::Detector det, int subdet, std::unique_ptr<const CaloSubdetectorTopology> geom) {
  int index=makeIndex(det,subdet);
  theTopologies_[index]=std::move(geom);
}

const CaloSubdetectorTopology* CaloTopology::getSubdetectorTopology(const DetId& id) const {
  auto i=theTopologies_.find(makeIndex(id.det(),id.subdetId()));
  return (i==theTopologies_.end())?(nullptr):(i->second.get());
}

const CaloSubdetectorTopology* CaloTopology::getSubdetectorTopology(DetId::Detector det, int subdet) const {
    auto i=theTopologies_.find(makeIndex(det,subdet));
    return (i==theTopologies_.end())?(nullptr):(i->second.get());
}

static const std::vector<DetId> emptyDetIdVector;

std::vector<DetId> CaloTopology::east(const DetId& id) const {
  const CaloSubdetectorTopology* topology=getSubdetectorTopology(id);
  return (topology==nullptr) ? (emptyDetIdVector):(topology->east(id));
}

std::vector<DetId> CaloTopology::west(const DetId& id) const {
  const CaloSubdetectorTopology* topology=getSubdetectorTopology(id);
  return (topology==nullptr) ? (emptyDetIdVector):(topology->west(id));
}

std::vector<DetId> CaloTopology::north(const DetId& id) const {
  const CaloSubdetectorTopology* topology=getSubdetectorTopology(id);
  return (topology==nullptr) ? (emptyDetIdVector):(topology->north(id));
}

std::vector<DetId> CaloTopology::south(const DetId& id) const {
  const CaloSubdetectorTopology* topology=getSubdetectorTopology(id);
  return (topology==nullptr) ? (emptyDetIdVector):(topology->south(id));
}

std::vector<DetId> CaloTopology::up(const DetId& id) const {
  const CaloSubdetectorTopology* topology=getSubdetectorTopology(id);
  return (topology==nullptr) ? (emptyDetIdVector):(topology->up(id));
}

std::vector<DetId> CaloTopology::down(const DetId& id) const {
    const CaloSubdetectorTopology* topology=getSubdetectorTopology(id);
    return (topology==nullptr) ? (emptyDetIdVector):(topology->down(id));
}

std::vector<DetId> CaloTopology::getNeighbours(const DetId& id,const CaloDirection& dir) const {
    const CaloSubdetectorTopology* topology=getSubdetectorTopology(id);
    return (topology==nullptr) ? (emptyDetIdVector):(topology->getNeighbours(id,dir));
}

std::vector<DetId> CaloTopology::getWindow(const DetId& id, const int& northSouthSize, const int& eastWestSize) const {
    const CaloSubdetectorTopology* topology=getSubdetectorTopology(id);
    return (topology==nullptr) ? (emptyDetIdVector):(topology->getWindow(id,northSouthSize, eastWestSize));
}

std::vector<DetId> CaloTopology::getAllNeighbours(const DetId& id) const {
    const CaloSubdetectorTopology* topology=getSubdetectorTopology(id);
    return (topology==nullptr) ? (emptyDetIdVector):(topology->getAllNeighbours(id));
}

bool CaloTopology::valid(const DetId& id) const {
  const CaloSubdetectorTopology* geom=getSubdetectorTopology(id);
  return (geom==nullptr)?(false):(geom->valid(id));
}

