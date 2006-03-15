#ifndef TRACKERLAYERIDACCESSOR_H
#define TRACKERLAYERIDACCESSOR_H


#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include<ext/functional>

using namespace std;

//class DetIdComparator : public binary_function<DetId, DetId, bool> {
class DetIdComparator {
public:
  virtual bool operator()( DetId i1, DetId i2 ) const {
    return i1<i2;
  }
};

class DetIdTIBSameLayerComparator : public  DetIdComparator {
 public:
  virtual bool operator()( DetId i1, DetId i2 ) const {
    TIBDetId id1(i1);
    TIBDetId id2(i2);
    if ((id1.det() == id2.det()) &&
	(id1.subdetId() == id2.subdetId()) &&
	(id1.layer() == id2.layer())) return false;
    return (id1<id2);
  }
};
class DetIdTOBSameLayerComparator : public  DetIdComparator {
 public:
  virtual bool operator()( DetId i1, DetId i2 ) const {
    TOBDetId id1(i1);
    TOBDetId id2(i2);
    if ((id1.det() == id2.det()) &&
	(id1.subdetId() == id2.subdetId()) &&
	(id1.layer() == id2.layer())) return false;
    return (id1<id2);
  }
};
class DetIdPXBSameLayerComparator : public  DetIdComparator {
 public:
  virtual bool operator()( DetId i1, DetId i2 ) const {
    PXBDetId id1(i1);
    PXBDetId id2(i2);
    if ((id1.det() == id2.det()) &&
	(id1.subdetId() == id2.subdetId()) &&
	(id1.layer() == id2.layer())) return false;
    return (id1<id2);
  }
};
class DetIdPXFSameDiskComparator : public  DetIdComparator {
 public:
  virtual bool operator()( DetId i1, DetId i2 ) const {
    PXFDetId id1(i1);
    PXFDetId id2(i2);
    if ((id1.det() == id2.det()) &&
	(id1.subdetId() == id2.subdetId()) &&
	(id1.side() == id2.side()) &&
	(id1.disk() == id2.disk())) return false;
    return (id1<id2);
  }
};
class DetIdTECSameDiskComparator : public  DetIdComparator {
 public:
  virtual bool operator()( DetId i1, DetId i2 ) const {
    TECDetId id1(i1);
    TECDetId id2(i2);
    if ((id1.det() == id2.det()) &&
	(id1.subdetId() == id2.subdetId()) &&
	(id1.side() == id2.side()) &&
	(id1.wheel() == id2.wheel())) return false;
    return (id1<id2);
  }
};
class DetIdTIDSameDiskComparator : public  DetIdComparator {
 public:
  virtual bool operator()( DetId i1, DetId i2 ) const {
    TIDDetId id1(i1);
    TIDDetId id2(i2);
    if ((id1.det() == id2.det()) &&
	(id1.subdetId() == id2.subdetId()) &&
	(id1.side() == id2.side()) &&
	(id1.wheel() == id2.wheel())) return false;
    return (id1<id2);
  }
};


class TrackerLayerIdAccessor {
 public:
  //
  // returns a valid DetId + a valid comaprator for the RangeMap
  //
  typedef std::pair<DetId,DetIdComparator> returnType;
  TrackerLayerIdAccessor();
  returnType pixelBarrelLayer(int layer);
  returnType pixelForwardDisk(int side,int disk);
  returnType stripTIBLayer(int layer);
  returnType stripTOBLayer(int layer);
  returnType stripTECDisk(int side,int disk);
  returnType stripTIDDisk(int side,int disk);
  
 private:
  
};

TrackerLayerIdAccessor::TrackerLayerIdAccessor(){}

TrackerLayerIdAccessor::returnType TrackerLayerIdAccessor::pixelBarrelLayer(int layer ){
  PXBDetId id(layer,1,1);
  return make_pair(id,DetIdPXBSameLayerComparator());
}
TrackerLayerIdAccessor::returnType TrackerLayerIdAccessor::pixelForwardDisk(int side, int disk ){
  PXFDetId id(side,disk,1,1,1);
  return make_pair(id,DetIdPXFSameDiskComparator());
}
TrackerLayerIdAccessor::returnType TrackerLayerIdAccessor::stripTIBLayer(int layer ){
  TIBDetId id(layer,1,1,1,1,1);
  return make_pair(id,DetIdTIBSameLayerComparator());
}
TrackerLayerIdAccessor::returnType TrackerLayerIdAccessor::stripTOBLayer(int layer ){
  TOBDetId id(layer,1,1,1,1);
  return make_pair(id,DetIdTOBSameLayerComparator());
}
TrackerLayerIdAccessor::returnType TrackerLayerIdAccessor::stripTIDDisk(int side, int disk ){
  TIDDetId id(side,disk,1,1,1,1);
  return make_pair(id,DetIdTIDSameDiskComparator());
}
TrackerLayerIdAccessor::returnType TrackerLayerIdAccessor::stripTECDisk(int side, int disk ){
  TECDetId id(side,disk,1,1,1,1,1,1);
  return make_pair(id,DetIdTECSameDiskComparator());
}

#endif

