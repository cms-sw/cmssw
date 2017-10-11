#include "Geometry/HGCalGeometry/interface/HGCellGeometry.h"
#include <algorithm>
#include <iostream>

//#define EDM_ML_DEBUG

HGCellGeometry::HGCellGeometry() : FlatTrd(),  topo_(nullptr) { }

HGCellGeometry::HGCellGeometry(const HGCellGeometry& tr) : FlatTrd(tr),
							   topo_(tr.topo_) {
  *this = tr ; 
}

HGCellGeometry& HGCellGeometry::operator=(const HGCellGeometry& tr) {
  FlatTrd::operator=( tr ) ;
  if ( this != &tr ) topo_ = tr.topo_;
#ifdef EDM_ML_DEBUG
  std::cout << "HGCellGeometry(Copy): topo_ " << topo_ << std::endl;
#endif
  return *this ; 
}

HGCellGeometry::HGCellGeometry(const HGCalTopology* topo, CornersMgr*  cMgr ,
			       const GlobalPoint& fCtr ,
			       const GlobalPoint& bCtr ,
			       const GlobalPoint& cor1 ,
			       const CCGFloat*    parV ) :
  FlatTrd(cMgr, fCtr, bCtr, cor1, parV), topo_(topo) {
#ifdef EDM_ML_DEBUG
  std::cout << "HGCellGeometry: topo_ " << topo_ << std::endl;
#endif
} 

HGCellGeometry::HGCellGeometry(const HGCalTopology* topo, 
			       const CornersVec& corn ,
			       const CCGFloat*   par    ) :
  FlatTrd(corn, par), topo_(topo) {
#ifdef EDM_ML_DEBUG
  std::cout << "HGCellGeometry: topo_ " << topo_ << std::endl;
#endif
} 

HGCellGeometry::HGCellGeometry(const HGCalTopology* topo, const FlatTrd& tr,
			       const Pt3D & local ) : 
  FlatTrd(tr, local), topo_(topo) {
#ifdef EDM_ML_DEBUG
  std::cout << "HGCellGeometry: topo_ " << topo_ << std::endl;
#endif
}

HGCellGeometry::~HGCellGeometry() {}

const GlobalPoint HGCellGeometry::getPosition(const DetId& id) const {

  HGCalTopology::DecodedDetId id_ = topology().decode(id);
  std::pair<float,float> xy;
  if (topology().dddConstants().geomMode() == HGCalGeometryMode::Square) {
    xy = topology().dddConstants().locateCell(id_.iCell,id_.iLay,id_.iSubSec,true);
  } else {
    xy = topology().dddConstants().locateCellHex(id_.iCell,id_.iSec,true);
  }
  const HepGeom::Point3D<float> lcoord(xy.first,xy.second,0);
  GlobalPoint glob = FlatTrd::getPosition(lcoord);
#ifdef EDM_ML_DEBUG
  std::cout << "getPosition:: Local " << lcoord.x() << ":" << lcoord.y() 
	    << " ID " << id_.iCell << ":" << id_.iLay  << " Global " << glob
	    << std::endl;
#endif
  return glob;
}

const std::vector<GlobalPoint> HGCellGeometry::getCorners(const DetId& id) const {

  std::vector<GlobalPoint> co (8, GlobalPoint(0,0,0));
  HGCalTopology::DecodedDetId id_ = topology().decode(id);
  std::pair<float,float> xy;
  if (topology().dddConstants().geomMode() == HGCalGeometryMode::Square) {
    xy = topology().dddConstants().locateCell(id_.iCell,id_.iLay,id_.iSubSec,true);
  } else {
    xy = topology().dddConstants().locateCellHex(id_.iCell,id_.iSec,true);
  }
  float dz = param()[0];
  float dx = param()[11];
  static const int signx[] = {-1,-1,1,1,-1,-1,1,1};
  static const int signy[] = {-1,1,1,-1,-1,1,1,-1};
  static const int signz[] = {-1,-1,-1,-1,1,1,1,1};
  for (unsigned int i = 0; i != 8; ++i) {
    const HepGeom::Point3D<float> lcoord(xy.first+signx[i]*dx,xy.second+signy[i]*dx,signz[i]*dz);
    co[i] = FlatTrd::getPosition(lcoord);
  }
  return co;
}

//----------------------------------------------------------------------

std::ostream& operator<<( std::ostream& s, const HGCellGeometry& cell0) {
  const FlatTrd& cell(cell0);
  s << "Center: " << cell.getPosition() << " eta " << cell.etaPos()
    << " phi " << cell.phiPos() << std::endl;
  s << "Axis: " << cell.getThetaAxis() << " " << cell.getPhiAxis() <<std::endl;
  const CaloCellGeometry::CornersVec& corners ( cell.getCorners() ) ;
  for ( unsigned int i=0 ; i != corners.size() ; ++i ) {
    s << "Corner: " << corners[i] << std::endl;
  }
  return s ;
}
  
