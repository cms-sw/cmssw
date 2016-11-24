#ifndef Geometry_CaloTopology_FastTimeTopology_h
#define Geometry_CaloTopology_FastTimeTopology_h 1

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"
#include <vector>
#include <iostream>

class FastTimeTopology : public CaloSubdetectorTopology {

public:
  /// create a new Topology
  FastTimeTopology(const FastTimeDDDConstants& hdcons, 
		   ForwardSubdetector subdet, int type);

  /// virtual destructor
  virtual ~FastTimeTopology() { }  

  /// move the Topology north (increment iy)  
  virtual DetId  goNorth(const DetId& id) const {
    return offsetBy(id,0,+1);
  }
  virtual std::vector<DetId> north(const DetId& id) const { 
    DetId nextId= goNorth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(nextId);
    return vNeighborsDetId;
  }

  /// move the Topology south (decrement iy)
  virtual DetId goSouth(const DetId& id) const {
    return offsetBy(id,0,-1);
  }
  virtual std::vector<DetId> south(const DetId& id) const { 
    DetId nextId= goSouth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(nextId);
    return vNeighborsDetId;
  }

  /// move the Topology east (positive ix)
  virtual DetId  goEast(const DetId& id) const {
    return offsetBy(id,+1,0);
  }
  virtual std::vector<DetId> east(const DetId& id) const { 
    DetId nextId=goEast(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(nextId);
    return vNeighborsDetId;
  }

  /// move the Topology west (negative ix)
  virtual DetId goWest(const DetId& id) const {
    return offsetBy(id,-1,0);
  }
  virtual std::vector<DetId> west(const DetId& id) const { 
    DetId nextId=goWest(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(nextId);
    return vNeighborsDetId;
  }
  
  virtual std::vector<DetId> up(const DetId& id) const {
    std::vector<DetId> vNeighborsDetId;
    return vNeighborsDetId;
  }
  
  virtual std::vector<DetId> down(const DetId& id) const {
    std::vector<DetId> vNeighborsDetId;
    return vNeighborsDetId;
  }

  ///Dense indexing
  virtual uint32_t detId2denseId(const DetId& id) const;
  virtual DetId    denseId2detId(uint32_t denseId) const;
  virtual uint32_t detId2denseGeomId(const DetId& id) const;

  ///Is this a valid cell id
  virtual bool valid(const DetId& id) const;
  bool validHashIndex(uint32_t ix) const {return (ix < kSizeForDenseIndexing);}

  unsigned int totalModules() const {return kSizeForDenseIndexing;}
  unsigned int totalGeomModules() const {return (unsigned int)(2*kHGeomHalf_);}
  int          numberCells() const {return kHGeomHalf_;}

  const FastTimeDDDConstants& dddConstants () const {return hdcons_;}

  DetId offsetBy(const DetId startId, int nrStepsX, int nrStepsY ) const;
  DetId switchZSide(const DetId startId) const;

  struct DecodedDetId {
    DecodedDetId() : iPhi(0), iEtaZ(0), zside(0), iType(0), subdet(0) {}
    int                       iPhi, iEtaZ, zside, iType, subdet;
  };

  DecodedDetId geomDenseId2decId(const uint32_t& hi) const;
  DecodedDetId decode(const DetId& id)  const ;
  DetId encode(const DecodedDetId& id_) const ;

  ForwardSubdetector subDetector()  const { return subdet_;}
  int                detectorType() const { return type_;}
private:

  /// move the nagivator along x, y
  DetId changeXY(const DetId& id, int nrStepsX, int nrStepsY) const ;

  const FastTimeDDDConstants& hdcons_;
  ForwardSubdetector          subdet_;
  int                         type_, nEtaZ_, nPhi_, kHGhalf_, kHGeomHalf_;
  unsigned int                kSizeForDenseIndexing;
};

#endif
