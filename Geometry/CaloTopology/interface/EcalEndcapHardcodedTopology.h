#ifndef GEOMETRY_CALOTOPOLOGY_ECALENDCAPHARDCODEDTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_ECALENDCAPHARDCODEDTOPOLOGY_H 1

#include <vector>
#include <iostream>
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

class EcalEndcapHardcodedTopology GCC11_FINAL : public CaloSubdetectorTopology
{

 public:
  /// create a new Topology
  EcalEndcapHardcodedTopology() {};

  virtual ~EcalEndcapHardcodedTopology() {};
  
   /// move the Topology north (increment iy)  
  virtual DetId  goNorth(const DetId& id) const {
    return incrementIy(EEDetId(id));
  }
  virtual std::vector<DetId> north(const DetId& id) const
  { 
    EEDetId nextId= goNorth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==EEDetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  /// move the Topology south (decrement iy)
  virtual DetId goSouth(const DetId& id) const {
    return decrementIy(EEDetId(id));
  }
  virtual std::vector<DetId> south(const DetId& id) const
  { 
    EEDetId nextId= goSouth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==EEDetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  /// move the Topology east (positive ix)
  virtual DetId  goEast(const DetId& id) const {
    return incrementIx(EEDetId(id));
  }
  virtual std::vector<DetId> east(const DetId& id) const
  { 
    EEDetId nextId=goEast(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==EEDetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  /// move the Topology west (negative ix)
  virtual DetId goWest(const DetId& id) const {
    return decrementIx(EEDetId(id));
  }
  virtual std::vector<DetId> west(const DetId& id) const
  { 
    EEDetId nextId=goWest(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==EEDetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  virtual std::vector<DetId> up(const DetId& /*id*/) const
  {
    std::cout << "EcalEndcapHardcodedTopology::up() not yet implemented" << std::endl; 
    std::vector<DetId> vNeighborsDetId;
    return  vNeighborsDetId;
  }
  
  virtual std::vector<DetId> down(const DetId& /*id*/) const
  {
    std::cout << "EcalEndcapHardcodedTopology::down() not yet implemented" << std::endl; 
    std::vector<DetId> vNeighborsDetId;
    return  vNeighborsDetId;
  }
  
 private:
  
  /// move the nagivator to larger ix
  EEDetId incrementIx(const EEDetId&) const ;

  /// move the nagivator to smaller ix
  EEDetId decrementIx(const EEDetId&) const ;

  /// move the nagivator to larger iy
  EEDetId incrementIy(const EEDetId&) const ;

  /// move the nagivator to smaller iy
  EEDetId decrementIy(const EEDetId&) const;

};

#endif
