/** Implementation of the Model for GEM Geometry
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

GEMGeometry::GEMGeometry() {}

GEMGeometry::~GEMGeometry() {}

const GEMGeometry::DetTypeContainer& GEMGeometry::detTypes() const { return theEtaPartitionTypes; }

const GEMGeometry::DetContainer& GEMGeometry::detUnits() const { return theEtaPartitions; }

const GEMGeometry::DetContainer& GEMGeometry::dets() const { return theDets; }

const GEMGeometry::DetIdContainer& GEMGeometry::detUnitIds() const { return theEtaPartitionIds; }

const GEMGeometry::DetIdContainer& GEMGeometry::detIds() const { return theDetIds; }

const GeomDet* GEMGeometry::idToDetUnit(DetId id) const { return dynamic_cast<const GeomDet*>(idToDet(id)); }

const GeomDet* GEMGeometry::idToDet(DetId id) const {
  mapIdToDet::const_iterator i = theMap.find(GEMDetId(id).rawId());
  return (i != theMap.end()) ? i->second : nullptr;
}

const std::vector<const GEMRegion*>& GEMGeometry::regions() const { return allRegions; }

const std::vector<const GEMStation*>& GEMGeometry::stations() const { return allStations; }

const std::vector<const GEMRing*>& GEMGeometry::rings() const { return allRings; }

const std::vector<const GEMSuperChamber*>& GEMGeometry::superChambers() const { return allSuperChambers; }

const std::vector<const GEMChamber*>& GEMGeometry::chambers() const { return allChambers; }

const std::vector<const GEMEtaPartition*>& GEMGeometry::etaPartitions() const { return allEtaPartitions; }

const GEMRegion* GEMGeometry::region(int re) const {
  for (auto region : allRegions) {
    if (re != region->region())
      continue;
    return region;
  }
  return nullptr;
}

const GEMStation* GEMGeometry::station(int re, int st) const {
  for (auto station : allStations) {
    if (re != station->region() || st != station->station())
      continue;
    return station;
  }
  return nullptr;
}

const GEMRing* GEMGeometry::ring(int re, int st, int ri) const {
  for (auto ring : allRings) {
    if (re != ring->region() || st != ring->station() || ri != ring->ring())
      continue;
    return ring;
  }
  return nullptr;
}

const GEMSuperChamber* GEMGeometry::superChamber(GEMDetId id) const {
  return dynamic_cast<const GEMSuperChamber*>(idToDet(id.superChamberId()));
}

const GEMChamber* GEMGeometry::chamber(GEMDetId id) const {
  return dynamic_cast<const GEMChamber*>(idToDet(id.chamberId()));
}

const GEMEtaPartition* GEMGeometry::etaPartition(GEMDetId id) const {
  return dynamic_cast<const GEMEtaPartition*>(idToDetUnit(id));
}

void GEMGeometry::add(const GEMRegion* region) { allRegions.emplace_back(region); }

void GEMGeometry::add(const GEMStation* station) { allStations.emplace_back(station); }

void GEMGeometry::add(const GEMRing* ring) { allRings.emplace_back(ring); }

void GEMGeometry::add(const GEMSuperChamber* superChamber) {
  allSuperChambers.emplace_back(superChamber);
  theDets.emplace_back(superChamber);
  theDetIds.emplace_back(superChamber->geographicalId());
  theMap.insert(std::make_pair((superChamber->geographicalId()).rawId(), superChamber));
}

void GEMGeometry::add(const GEMEtaPartition* etaPartition) {
  theDets.emplace_back(etaPartition);
  allEtaPartitions.emplace_back(etaPartition);
  theEtaPartitions.emplace_back(etaPartition);
  theEtaPartitionIds.emplace_back(etaPartition->geographicalId());
  theDetIds.emplace_back(etaPartition->geographicalId());
  theEtaPartitionTypes.emplace_back(&etaPartition->type());
  theMap.insert(std::make_pair((etaPartition->geographicalId()).rawId(), etaPartition));
}

void GEMGeometry::add(const GEMChamber* chamber) {
  allChambers.emplace_back(chamber);
  theDets.emplace_back(chamber);
  theDetIds.emplace_back(chamber->geographicalId());
  theMap.insert(std::make_pair((chamber->geographicalId()).rawId(), chamber));
}

bool GEMGeometry::hasME0() const { return station(1, 0) != nullptr and station(-1, 0) != nullptr; }

bool GEMGeometry::hasGE11() const { return station(1, 1) != nullptr and station(-1, 1) != nullptr; }

bool GEMGeometry::hasGE21() const { return station(1, 2) != nullptr and station(-1, 2) != nullptr; }
