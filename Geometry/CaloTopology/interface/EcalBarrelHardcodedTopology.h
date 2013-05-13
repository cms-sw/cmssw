#ifndef GEOMETRY_CALOTOPOLOGY_ECALBARRELHARDCODEDTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_ECALBARRELHARDCODEDTOPOLOGY_H 1

#include <vector>
#include <iostream>
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

class EcalBarrelHardcodedTopology GCC11_FINAL : public CaloSubdetectorTopology
{

 public:
  /// create a new Topology
  EcalBarrelHardcodedTopology() {};

  virtual ~EcalBarrelHardcodedTopology() {};
  
  /// move the Topology north (increment iphi)
  virtual DetId  goNorth(const DetId& id) const {
    return incrementIphi(EBDetId(id));
  }
  virtual std::vector<DetId> north(const DetId& id) const
    { 
      EBDetId nextId=goNorth(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==EBDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology south (decrement iphi)
  virtual DetId goSouth(const DetId& id) const {
    return decrementIphi(EBDetId(id));
  }
  virtual std::vector<DetId> south(const DetId& id) const
    { 
      EBDetId nextId=goSouth(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==EBDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology east (negative ieta)
  virtual DetId  goEast(const DetId& id) const {
    return decrementIeta(EBDetId(id));
  }
  virtual std::vector<DetId> east(const DetId& id) const
    { 
      EBDetId nextId=goEast(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==EBDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology west (positive ieta)
  virtual DetId  goWest(const DetId& id) const {
    return incrementIeta(EBDetId(id));
  }
  virtual std::vector<DetId> west(const DetId& id) const
    { 
      EBDetId nextId=goWest(id);
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==EBDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }
  
  virtual std::vector<DetId> up(const DetId& /*id*/) const
    {
      std::cout << "EcalBarrelHardcodedTopology::up() not yet implemented" << std::endl; 
      std::vector<DetId> vNeighborsDetId;
      return  vNeighborsDetId;
    }
  
  virtual std::vector<DetId> down(const DetId& /*id*/) const
    {
      std::cout << "EcalBarrelHardcodedTopology::down() not yet implemented" << std::endl; 
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

};

#endif
