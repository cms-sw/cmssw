#ifndef GEOMETRY_CALOTOPOLOGY_ECALBARRELTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_ECALBARRELTOPOLOGY_H 1

#include <utility>
#include <vector>
#include <iostream>
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

class EcalBarrelTopology final : public CaloSubdetectorTopology
{

 public:
  /// create a new Topology
  EcalBarrelTopology(): theGeom_(nullptr) {};

  /// virtual destructor
  ~EcalBarrelTopology() override { }  

  /// create a new Topology from geometry
  EcalBarrelTopology(edm::ESHandle<CaloGeometry> theGeom) : theGeom_(std::move(theGeom))
    {
    }
  
 /// move the Topology north (increment iphi)
  DetId  goNorth(const DetId& id) const override {
    return incrementIphi(EBDetId(id));
  }
  std::vector<DetId> north(const DetId& id) const override
    { 
      EBDetId nextId=goNorth(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==EBDetId(0)))
	vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology south (decrement iphi)
  DetId goSouth(const DetId& id) const override {
    return decrementIphi(EBDetId(id));
  }
  std::vector<DetId> south(const DetId& id) const override
    { 
      EBDetId nextId=goSouth(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==EBDetId(0)))
	vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology east (negative ieta)
  DetId  goEast(const DetId& id) const override {
    return decrementIeta(EBDetId(id));
  }
  std::vector<DetId> east(const DetId& id) const override
    { 
      EBDetId nextId=goEast(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==EBDetId(0)))
	vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology west (positive ieta)
  DetId  goWest(const DetId& id) const override {
    return incrementIeta(EBDetId(id));
  }
  std::vector<DetId> west(const DetId& id) const override
    { 
      EBDetId nextId=goWest(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==EBDetId(0)))
	vNeighborsDetId.emplace_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }
  
  
  std::vector<DetId> up(const DetId& /*id*/) const override
    {
      std::cout << "EcalBarrelTopology::up() not yet implemented" << std::endl; 
      std::vector<DetId> vNeighborsDetId;
      return  vNeighborsDetId;
    }
  
  std::vector<DetId> down(const DetId& /*id*/) const override
    {
      std::cout << "EcalBarrelTopology::down() not yet implemented" << std::endl; 
      std::vector<DetId> vNeighborsDetId;
      return  vNeighborsDetId;
    }

 private:

  /// move the nagivator to larger ieta (more positive z) (stops at end of barrel and returns null)
  EBDetId incrementIeta(const EBDetId&) const ;

  /// move the nagivator to smaller ieta (more negative z) (stops at end of barrel and returns null)
  EBDetId decrementIeta(const EBDetId&) const ;

  /// move the nagivator to larger iphi (wraps around the barrel) 
  EBDetId incrementIphi(const EBDetId&) const ;

  /// move the nagivator to smaller iphi (wraps around the barrel)
  EBDetId decrementIphi(const EBDetId&) const;

  edm::ESHandle<CaloGeometry> theGeom_;
};

#endif
