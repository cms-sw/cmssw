#ifndef Geometry_CaloTopology_HGCalTopology_h
#define Geometry_CaloTopology_HGCalTopology_h 1

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include <vector>
#include <iostream>

class HGCalTopology : public CaloSubdetectorTopology {

public:
  /// create a new Topology
  HGCalTopology(const HGCalDDDConstants& hdcons, int subdet);

  /// virtual destructor
  ~HGCalTopology() override { }  

  /// move the Topology north (increment iy)  
  DetId  goNorth(const DetId& id) const override {
    return changeXY(id,0,+1);
  }
  std::vector<DetId> north(const DetId& id) const override { 
    DetId nextId= goNorth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.emplace_back(nextId);
    return vNeighborsDetId;
  }

  /// move the Topology south (decrement iy)
  DetId goSouth(const DetId& id) const override {
    return changeXY(id,0,-1);
  }
  std::vector<DetId> south(const DetId& id) const override { 
    DetId nextId= goSouth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.emplace_back(nextId);
    return vNeighborsDetId;
  }

  /// move the Topology east (positive ix)
  DetId  goEast(const DetId& id) const override {
    return changeXY(id,+1,0);
  }
  std::vector<DetId> east(const DetId& id) const override { 
    DetId nextId=goEast(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.emplace_back(nextId);
    return vNeighborsDetId;
  }

  /// move the Topology west (negative ix)
  DetId goWest(const DetId& id) const override {
    return changeXY(id,-1,0);
  }
  std::vector<DetId> west(const DetId& id) const override { 
    DetId nextId=goWest(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.emplace_back(nextId);
    return vNeighborsDetId;
  }
  
  std::vector<DetId> up(const DetId& id) const override {
    DetId nextId=changeZ(id,+1);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.emplace_back(nextId);
    return vNeighborsDetId;
  }
  
  std::vector<DetId> down(const DetId& id) const override {
    DetId nextId=changeZ(id,-1);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==DetId(0)))
      vNeighborsDetId.emplace_back(nextId);
    return vNeighborsDetId;
  }

  std::vector<DetId> neighbors(const DetId& id) const;

  ///Geometry mode
  HGCalGeometryMode::GeometryMode geomMode() const {return mode_;}

  ///Dense indexing
  uint32_t detId2denseId(const DetId& id) const override;
  DetId denseId2detId(uint32_t denseId) const override;
  virtual uint32_t detId2denseGeomId(const DetId& id) const;

  ///Is this a valid cell id
  bool valid(const DetId& id) const override;
  bool validHashIndex(uint32_t ix) const {return (ix < kSizeForDenseIndexing);}

  unsigned int totalModules() const {return kSizeForDenseIndexing;}
  unsigned int totalGeomModules() const {return (unsigned int)(2*kHGeomHalf_);}
  unsigned int allGeomModules() const;

  const HGCalDDDConstants& dddConstants () const {return hdcons_;}
  
  /** returns a new DetId offset by nrStepsX and nrStepsY (can be negative),
   * returns DetId(0) if invalid */
  DetId offsetBy(const DetId startId, int nrStepsX, int nrStepsY) const;
  DetId switchZSide(const DetId startId) const;

  /// Use subSector in square mode as wafer type in hexagon mode
  static const int subSectors_ = 2;

  struct DecodedDetId {
    DecodedDetId() : iCell1(0), iCell2(0), iLay(0), iSec1(0), iSec2(0),
                     iType(0), zSide(0), det(0) {}
    int              iCell1, iCell2, iLay, iSec1, iSec2, iType, zSide, det;
  };

  DecodedDetId geomDenseId2decId(const uint32_t& hi) const;
  DecodedDetId decode(const DetId& id)  const ;
  DetId encode(const DecodedDetId& id_) const ;

  DetId::Detector    detector()  const { return det_;}
  ForwardSubdetector subDetector()  const {return subdet_;}
  bool               detectorType() const { return false;}
  bool               isHFNose() const { 
    return (((det_ == DetId::Forward) && 
	     (subdet_ == ForwardSubdetector::HFNose)) ?  true : false);
  }
private:
  /// add DetId of Scintillator and Silicon type if valid
  void addHGCSCintillatorId(std::vector<DetId>& ids, int zside, int type,
			    int lay, int iradius, int iphi) const;
  void addHGCSiliconId(std::vector<DetId>& ids, int det, int zside,
		       int type,  int lay, int waferU, int waferV, 
		       int cellU, int cellV) const;

  /// move the nagivator along x, y
  DetId changeXY(const DetId& id, int nrStepsX, int nrStepsY) const ;

  /// move the nagivator along z
  DetId changeZ(const DetId& id, int nrStepsZ) const ;

  const HGCalDDDConstants&        hdcons_;
  HGCalGeometryMode::GeometryMode mode_;

  DetId::Detector                 det_;
  ForwardSubdetector              subdet_;
  int                             sectors_, layers_, cells_, types_;
  int                             firstLay_, cellMax_, waferOff_, waferMax_;
  int                             kHGhalf_, kHGeomHalf_;
  unsigned int                    kSizeForDenseIndexing;
};

#endif
