#ifndef Geometry_CaloTopology_HGCalTopology_h
#define Geometry_CaloTopology_HGCalTopology_h 1

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include <vector>
#include <iostream>

class HGCalTopology : public CaloSubdetectorTopology {

public:
  /// create a new Topology
  HGCalTopology(const HGCalDDDConstants& hdcons, ForwardSubdetector subdet, bool halfChamber);

  /// virtual destructor
  virtual ~HGCalTopology() { }  

  /// move the Topology north (increment iy)  
  virtual DetId  goNorth(const DetId& id) const {
    return changeXY(id,0,+1);
  }
  virtual std::vector<DetId> north(const DetId& id) const { 
    DetId nextId= goNorth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }

  /// move the Topology south (decrement iy)
  virtual DetId goSouth(const DetId& id) const {
    return changeXY(id,0,-1);
  }
  virtual std::vector<DetId> south(const DetId& id) const { 
    DetId nextId= goSouth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }

  /// move the Topology east (positive ix)
  virtual DetId  goEast(const DetId& id) const {
    return changeXY(id,+1,0);
  }
  virtual std::vector<DetId> east(const DetId& id) const { 
    DetId nextId=goEast(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }

  /// move the Topology west (negative ix)
  virtual DetId goWest(const DetId& id) const {
    return changeXY(id,-1,0);
  }
  virtual std::vector<DetId> west(const DetId& id) const { 
    DetId nextId=goWest(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  virtual std::vector<DetId> up(const DetId& id) const {
    DetId nextId=changeZ(id,+1);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  virtual std::vector<DetId> down(const DetId& id) const {
    DetId nextId=changeZ(id,-1);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }

  ///Dense indexing
  virtual uint32_t detId2denseId(const DetId& id) const;
  virtual DetId denseId2detId(uint32_t denseId) const;
  virtual uint32_t detId2denseGeomId(const DetId& id) const;

  ///Is this a valid cell id
  virtual bool valid(const DetId& id) const;
  bool validHashIndex(uint32_t ix) const {return (ix < kSizeForDenseIndexing);}

  unsigned int totalModules() const {return kSizeForDenseIndexing;}
  unsigned int totalGeomModules() const {return (unsigned int)(2*kHGeomHalf_);}

  const HGCalDDDConstants& dddConstants () const {return hdcons_;}
  
  /** returns a new DetId offset by nrStepsX and nrStepsY (can be negative),
   * returns DetId(0) if invalid */
  DetId offsetBy(const DetId startId, int nrStepsX, int nrStepsY) const;
  DetId switchZSide(const DetId startId) const;

  static const int subSectors_ = 2;

  struct DecodedDetId {
    DecodedDetId() : iCell(0), iLay(0), iSec(0), iSubSec(0), zside(0), 
		     subdet(0) {}
    int                       iCell, iLay, iSec, iSubSec, zside, subdet;
  };

  DecodedDetId geomDenseId2decId(const uint32_t& hi) const;
  DecodedDetId decode(const DetId& id)  const ;
  DetId encode(const DecodedDetId& id_) const ;

  ForwardSubdetector subDetector()  const { return subdet_;}
  bool               detectorType() const { return half_;}
private:

  /// move the nagivator along x, y
  DetId changeXY(const DetId& id, int nrStepsX, int nrStepsY) const ;

  /// move the nagivator along z
  DetId changeZ(const DetId& id, int nrStepsZ) const ;

  const HGCalDDDConstants&    hdcons_;
  ForwardSubdetector          subdet_;
  bool                        half_;
  int                         sectors_, layers_, cells_, kHGhalf_, kHGeomHalf_;
  std::vector<int>            maxcells_;
  unsigned int                kSizeForDenseIndexing;
};

#endif
