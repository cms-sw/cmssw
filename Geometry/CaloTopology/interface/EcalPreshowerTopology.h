#ifndef GEOMETRY_CALOTOPOLOGY_ECALPRESHOWERTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_ECALPRESHOWERTOPOLOGY_H 1

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>
#include <iostream>

class EcalPreshowerTopology GCC11_FINAL : public CaloSubdetectorTopology {

 public:
  /// create a new Topology
  EcalPreshowerTopology() : theGeom_(0) {};

  /// virtual destructor
  virtual ~EcalPreshowerTopology() { }  
  
  /// create a new Topology from geometry
  EcalPreshowerTopology(edm::ESHandle<CaloGeometry> theGeom) : theGeom_(theGeom)
    {
    }

  /// move the Topology north (increment iy)
  virtual std::vector<DetId> north(const DetId& id) const
    { 
      ESDetId nextId=incrementIy(ESDetId(id));
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==ESDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology south (decrement iy)
  virtual std::vector<DetId> south(const DetId& id) const
    { 
      ESDetId nextId=decrementIy(ESDetId(id));
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==ESDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology east (positive ix)
  virtual std::vector<DetId> east(const DetId& id) const
    { 
      ESDetId nextId=incrementIx(ESDetId(id));
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==ESDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }

  /// move the Topology west (negative ix)
  virtual std::vector<DetId> west(const DetId& id) const
    { 
      ESDetId nextId=decrementIx(ESDetId(id));
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==ESDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
      return vNeighborsDetId;
    }
  
  virtual std::vector<DetId> up(const DetId& id) const
    {
      ESDetId nextId=incrementIz(ESDetId(id));
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==ESDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
      return  vNeighborsDetId;
    }
  
  virtual std::vector<DetId> down(const DetId& id) const
    {
      ESDetId nextId=decrementIz(ESDetId(id));
      std::vector<DetId> vNeighborsDetId;
      if (! (nextId==ESDetId(0)))
	vNeighborsDetId.push_back(DetId(nextId.rawId()));
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






