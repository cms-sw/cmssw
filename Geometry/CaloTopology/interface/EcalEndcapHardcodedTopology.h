#ifndef GEOMETRY_CALOTOPOLOGY_ECALENDCAPHARDCODEDTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_ECALENDCAPHARDCODEDTOPOLOGY_H 1

#include <vector>
#include <iostream>
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

class EcalEndcapHardcodedTopology final : public CaloSubdetectorTopology {
public:
  /// create a new Topology
  EcalEndcapHardcodedTopology(){};

  ~EcalEndcapHardcodedTopology() override{};

  /// move the Topology north (increment iy)
  DetId goNorth(const DetId& id) const override { return incrementIy(EEDetId(id)); }
  std::vector<DetId> north(const DetId& id) const override {
    EEDetId nextId = goNorth(id);
    std::vector<DetId> vNeighborsDetId;
    if (!(nextId == EEDetId(0)))
      vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }

  /// move the Topology south (decrement iy)
  DetId goSouth(const DetId& id) const override { return decrementIy(EEDetId(id)); }
  std::vector<DetId> south(const DetId& id) const override {
    EEDetId nextId = goSouth(id);
    std::vector<DetId> vNeighborsDetId;
    if (!(nextId == EEDetId(0)))
      vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }

  /// move the Topology east (positive ix)
  DetId goEast(const DetId& id) const override { return incrementIx(EEDetId(id)); }
  std::vector<DetId> east(const DetId& id) const override {
    EEDetId nextId = goEast(id);
    std::vector<DetId> vNeighborsDetId;
    if (!(nextId == EEDetId(0)))
      vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }

  /// move the Topology west (negative ix)
  DetId goWest(const DetId& id) const override { return decrementIx(EEDetId(id)); }
  std::vector<DetId> west(const DetId& id) const override {
    EEDetId nextId = goWest(id);
    std::vector<DetId> vNeighborsDetId;
    if (!(nextId == EEDetId(0)))
      vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }

  std::vector<DetId> up(const DetId& /*id*/) const override {
    edm::LogVerbatim("CaloTopology") << "EcalEndcapHardcodedTopology::up() not yet implemented";
    std::vector<DetId> vNeighborsDetId;
    return vNeighborsDetId;
  }

  std::vector<DetId> down(const DetId& /*id*/) const override {
    edm::LogVerbatim("CaloTopology") << "EcalEndcapHardcodedTopology::down() not yet implemented";
    std::vector<DetId> vNeighborsDetId;
    return vNeighborsDetId;
  }

private:
  /// move the nagivator to larger ix
  EEDetId incrementIx(const EEDetId&) const;

  /// move the nagivator to smaller ix
  EEDetId decrementIx(const EEDetId&) const;

  /// move the nagivator to larger iy
  EEDetId incrementIy(const EEDetId&) const;

  /// move the nagivator to smaller iy
  EEDetId decrementIy(const EEDetId&) const;
};

#endif
