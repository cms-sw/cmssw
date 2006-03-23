#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"


TrackerLayerIdAccessor::TrackerLayerIdAccessor(){}

std::pair<DetId,DetIdPXBSameLayerComparator> TrackerLayerIdAccessor::pixelBarrelLayer(int layer ){
  PXBDetId id(layer,1,1);
  return make_pair(id,DetIdPXBSameLayerComparator());
}
std::pair<DetId,DetIdPXFSameDiskComparator>  TrackerLayerIdAccessor::pixelForwardDisk(int side, int disk ){
  PXFDetId id(side,disk,1,1,1);
  return make_pair(id,DetIdPXFSameDiskComparator());
}
std::pair<DetId,DetIdTIBSameLayerComparator> TrackerLayerIdAccessor::stripTIBLayer(int layer ){
  TIBDetId id(layer,1,1,1,1,1);
  return make_pair(id,DetIdTIBSameLayerComparator());
}
std::pair<DetId,DetIdTOBSameLayerComparator> TrackerLayerIdAccessor::stripTOBLayer(int layer ){
  TOBDetId id(layer,1,1,1,1);
  return make_pair(id,DetIdTOBSameLayerComparator());
}
std::pair<DetId,DetIdTIDSameDiskComparator> TrackerLayerIdAccessor::stripTIDDisk(int side, int disk ){
  TIDDetId id(side,disk,1,1,1,1);
  return make_pair(id,DetIdTIDSameDiskComparator());
}
std::pair<DetId,DetIdTECSameDiskComparator> TrackerLayerIdAccessor::stripTECDisk(int side, int disk ){
  TECDetId id(side,disk,1,1,1,1,1);
  
  return make_pair(id,DetIdTECSameDiskComparator());
}
