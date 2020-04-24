#ifndef GEOMETRY_CALOTOPOLOGY_ECALPRESHOWERTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_ECALPRESHOWERTOPOLOGY_H 1

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <utility>
#include <vector>
#include <iostream>

class EcalPreshowerTopology final : public CaloSubdetectorTopology {

 public:
  /// create a new Topology
  EcalPreshowerTopology() : theGeom_(nullptr) {};

  /// virtual destructor
  ~EcalPreshowerTopology() override { }  
  
  /// create a new Topology from geometry
  EcalPreshowerTopology(edm::ESHandle<CaloGeometry> theGeom) : theGeom_(std::move(theGeom))
    {
    }

  
  /// move the Topology north (increment iy)  
  DetId  goNorth(const DetId& id) const override {
    return incrementIy(ESDetId(id));
  }
  std::vector<DetId> north(const DetId& id) const override
    { 
      ESDetId nextId= goNorth(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==ESDetId(0)))
	vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology south (decrement iy)
  DetId goSouth(const DetId& id) const override {
    return decrementIy(ESDetId(id));
  }
  std::vector<DetId> south(const DetId& id) const override
    { 
      ESDetId nextId= goSouth(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==ESDetId(0)))
	vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology east (positive ix)
  DetId  goEast(const DetId& id) const override {
    return incrementIx(ESDetId(id));
  }
  std::vector<DetId> east(const DetId& id) const override
  { 
    ESDetId nextId=goEast(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==ESDetId(0)))
      vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }

  /// move the Topology west (negative ix)
  DetId goWest(const DetId& id) const override {
    return decrementIx(ESDetId(id));
  }
  std::vector<DetId> west(const DetId& id) const override
  { 
    ESDetId nextId=goWest(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==ESDetId(0)))
      vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  DetId goUp(const DetId& id) const override {
    return incrementIz(ESDetId(id));
  }
  std::vector<DetId> up(const DetId& id) const override
  {
    ESDetId nextId=goUp(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==ESDetId(0)))
      vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
    return  vNeighborsDetId;
  }
  
  DetId goDown(const DetId& id) const override {
    return decrementIz(ESDetId(id));
  }
  std::vector<DetId> down(const DetId& id) const override
  {
    ESDetId nextId=goDown(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==ESDetId(0)))
      vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
    return  vNeighborsDetId;
  }

 private:

  /// move the nagivator to larger ix
  ESDetId incrementIx(const ESDetId& id) const ;

  /// move the nagivator to smaller ix
  ESDetId decrementIx(const ESDetId& id) const ;

  /// move the nagivator to larger iy
  ESDetId incrementIy(const ESDetId& id) const ;

  /// move the nagivator to smaller iy
  ESDetId decrementIy(const ESDetId& id) const;

  /// move the nagivator to larger iz
  ESDetId incrementIz(const ESDetId& id) const;

  /// move the nagivator to smaller iz
  ESDetId decrementIz(const ESDetId& id) const;

  edm::ESHandle<CaloGeometry> theGeom_;
};

#endif






