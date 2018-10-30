/* for High Granularity Calorimeter
 * This geometry is essentially driven by topology, 
 * which is thus encapsulated in this class. 
 * This makes this geometry not suitable to be loaded
 * by regular CaloGeometryLoader<T>
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cmath>

#include <Math/Transform3D.h>
#include <Math/EulerAngles.h>

typedef CaloCellGeometry::Tr3D Tr3D;
typedef std::vector<float> ParmVec;

//#define EDM_ML_DEBUG

HGCalGeometry::HGCalGeometry( const HGCalTopology& topology_ )
  : m_topology( topology_ ),
    m_validGeomIds( topology_.totalGeomModules()),
    mode_( topology_.geomMode()),
    m_det( topology_.detector()),
    m_subdet( topology_.subDetector()),
    twoBysqrt3_(2.0/std::sqrt(3.0)) {
  
  if (m_det == DetId::HGCalHSc) {
    m_cellVec2 = CellVec2(topology_.totalGeomModules());
  } else {
    m_cellVec = CellVec(topology_.totalGeomModules());
  }
  m_validIds.reserve( m_topology.totalModules());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Expected total # of Geometry Modules " 
				<< m_topology.totalGeomModules();
#endif
}

HGCalGeometry::~HGCalGeometry() { }

void HGCalGeometry::fillNamedParams (DDFilteredView fv) {}

void HGCalGeometry::initializeParms() {
}

void HGCalGeometry::localCorners(Pt3DVec&        lc,
				 const CCGFloat* pv,
				 unsigned int    i,
				 Pt3D&           ref) {
  if (m_det == DetId::HGCalHSc) {
    FlatTrd::localCorners( lc, pv, ref ) ;
  } else {
    FlatHexagon::localCorners( lc, pv, ref ) ;
  }
}

void HGCalGeometry::newCell( const GlobalPoint& f1 ,
			     const GlobalPoint& f2 ,
			     const GlobalPoint& f3 ,
			     const CCGFloat*    parm ,
			     const DetId&       detId ) {

  DetId geomId = getGeometryDetId(detId);
  int   cells (0);
  HGCalTopology::DecodedDetId id = m_topology.decode(detId);
  if ((mode_ == HGCalGeometryMode::Hexagon) || 
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    cells = m_topology.dddConstants().numberCellsHexagon(id.iSec1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "NewCell " << HGCalDetId(detId) 
				  << " GEOM " << HGCalDetId(geomId);
#endif
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    cells  = 1;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "NewCell " << HGCScintillatorDetId(detId) 
				  << " GEOM " << HGCScintillatorDetId(geomId);
#endif
  } else {
    cells  = m_topology.dddConstants().numberCellsHexagon(id.iLay,id.iSec1,
							  id.iSec2,false);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "NewCell " << HGCSiliconDetId(detId) 
				  << " GEOM " << HGCSiliconDetId(geomId);
#endif
  }
  const uint32_t cellIndex (m_topology.detId2denseGeomId(geomId));

  if (m_det == DetId::HGCalHSc) {
    m_cellVec2.at( cellIndex ) = FlatTrd( cornersMgr(), f1, f2, f3, parm ) ;
  } else {
    m_cellVec.at( cellIndex ) = FlatHexagon( cornersMgr(), f1, f2, f3, parm ) ;
  }
  m_validGeomIds.at( cellIndex ) = geomId ;
  
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Store for DetId " << std::hex 
				<< detId.rawId() << " GeomId " 
				<< geomId.rawId() << std::dec << " Index " 
				<< cellIndex << " cells " << cells;
  unsigned int nOld = m_validIds.size();
#endif
  if ((mode_ == HGCalGeometryMode::Hexagon) || 
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    for (int cell = 0; cell < cells; ++cell) {
      id.iCell1 = cell;
      DetId idc = m_topology.encode(id);
      if (m_topology.valid(idc)) {
	m_validIds.emplace_back(idc);
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("HGCalGeom") << "Valid Id [" << cell << "] "
				      << HGCalDetId(idc);
#endif
      }
    }
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    DetId idc = m_topology.encode(id);
    if (m_topology.valid(idc)) {
      m_validIds.emplace_back(idc);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Valid Id [0] " 
				    << HGCScintillatorDetId(idc);
#endif
    } else {
      edm::LogWarning("HGCalGeom") << "Check " << HGCScintillatorDetId(idc) 
				   << " from " << HGCScintillatorDetId(detId)
				   << " ERROR ???";
    }
  } else {
#ifdef EDM_ML_DEBUG
    unsigned int cellAll(0), cellSelect(0);
#endif
    for (int u=0; u<2*cells; ++u) {
      for (int v=0; v<2*cells; ++v) {
	if (((v-u) < cells) && (u-v) <= cells) {
	  id.iCell1 = u; id.iCell2 = v;
	  DetId idc = m_topology.encode(id);
#ifdef EDM_ML_DEBUG
	  ++cellAll;
#endif
	  if (m_topology.dddConstants().cellInLayer(id.iSec1,id.iSec2,u,v,
						    id.iLay,true)) {
	    m_validIds.emplace_back(idc);
#ifdef EDM_ML_DEBUG
	    ++cellSelect;
	    edm::LogVerbatim("HGCalGeom") << "Valid Id [" << u << ", " << v
					  << "] " << HGCSiliconDetId(idc);
#endif
	  }
	}
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "HGCalGeometry keeps " << cellSelect
				  << " out of " << cellAll << " for wafer "
				  << id.iSec1 << ":" << id.iSec2 << " in "
				  << " layer " << id.iLay;
#endif
  }
#ifdef EDM_ML_DEBUG
  if (m_det == DetId::HGCalHSc) {
    edm::LogVerbatim("HGCalGeom") << "HGCalGeometry::newCell-> [" << cellIndex 
				  << "]" << " front:" << f1.x() << '/' 
				  << f1.y() << '/' << f1.z() << " back:" 
				  << f2.x() << '/' << f2.y() << '/' << f2.z() 
				  << " eta|phi " 
				  << m_cellVec2[cellIndex].etaPos() << ":"
				  << m_cellVec2[cellIndex].phiPos();
  } else {
    edm::LogVerbatim("HGCalGeom") << "HGCalGeometry::newCell-> [" << cellIndex 
				  << "]" << " front:" << f1.x() << '/' 
				  << f1.y() << '/' << f1.z() << " back:" 
				  << f2.x() << '/' << f2.y() << '/' << f2.z() 
				  << " eta|phi " 
				  << m_cellVec[cellIndex].etaPos() << ":"
				  << m_cellVec[cellIndex].phiPos();
  }
  unsigned int nNew = m_validIds.size();
  if ((mode_ == HGCalGeometryMode::Hexagon) || 
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    edm::LogVerbatim("HGCalGeom") << "ID: " << HGCalDetId(detId) 
				  << " with valid DetId from " << nOld 
				  << " to " << nNew;
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    edm::LogVerbatim("HGCalGeom") << "ID: " << HGCScintillatorDetId(detId) 
				  << " with valid DetId from " << nOld 
				  << " to " << nNew;
  } else if (m_topology.isHFNose()) {
    edm::LogVerbatim("HGCalGeom") << "ID: " << HFNoseDetId(detId) 
				  << " with valid DetId from " << nOld 
				  << " to " << nNew;
  } else {
    edm::LogVerbatim("HGCalGeom") << "ID: " << HGCSiliconDetId(detId) 
				  << " with valid DetId from " << nOld 
				  << " to " << nNew;
  }
  edm::LogVerbatim("HGCalGeom") << "Cell[" << cellIndex << "] " << std::hex 
				<< geomId.rawId() << ":"  
				<< m_validGeomIds[cellIndex].rawId() 
				<< std::dec;
#endif
}

std::shared_ptr<const CaloCellGeometry> HGCalGeometry::getGeometry(const DetId& detId) const {
  if (detId == DetId()) return nullptr; // nothing to get
  DetId geomId = getGeometryDetId(detId);
  const uint32_t cellIndex (m_topology.detId2denseGeomId(geomId));
  const GlobalPoint pos = (detId != geomId) ? getPosition(detId) : GlobalPoint();
  return cellGeomPtr (cellIndex, pos);

}

bool HGCalGeometry::present(const DetId& detId) const {
  if (detId == DetId()) return false;
  DetId geomId = getGeometryDetId(detId);
  const uint32_t index (m_topology.detId2denseGeomId(geomId));
  return (nullptr != getGeometryRawPtr(index)) ;
}

GlobalPoint HGCalGeometry::getPosition(const DetId& id) const {

  unsigned int cellIndex =  indexFor(id);
  GlobalPoint glob;
  unsigned int maxSize = ((mode_ == HGCalGeometryMode::Trapezoid) ? 
			  m_cellVec2.size() : m_cellVec.size());
  if (cellIndex <  maxSize) {
    HGCalTopology::DecodedDetId id_ = m_topology.decode(id);
    std::pair<float,float> xy;
    if ((mode_ == HGCalGeometryMode::Hexagon) ||
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      xy = m_topology.dddConstants().locateCellHex(id_.iCell1,id_.iSec1,true);
      const HepGeom::Point3D<float> lcoord(xy.first,xy.second,0);
      glob = m_cellVec[cellIndex].getPosition(lcoord);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "getPosition:: index " << cellIndex 
				    << " Local " << lcoord.x() << ":" 
				    << lcoord.y() << " ID " << id_.iCell1 
				    << ":" << id_.iSec1 << " Global " << glob;
#endif
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      const HepGeom::Point3D<float> lcoord(0,0,0);
      glob = m_cellVec2[cellIndex].getPosition(lcoord);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "getPositionTrap:: index " << cellIndex 
				    << " Local " << lcoord.x() << ":" 
				    << lcoord.y() << " ID " << id_.iLay << ":"
				    << id_.iSec1 << ":" << id_.iCell1 
				    << " Global " << glob;
#endif
    } else {
      xy = m_topology.dddConstants().locateCell(id_.iLay,id_.iSec1,id_.iSec2,
						id_.iCell1,id_.iCell2,true,false);
      const HepGeom::Point3D<float> lcoord(xy.first,xy.second,0);
      glob = m_cellVec[cellIndex].getPosition(lcoord);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "getPositionWafer:: index " << cellIndex
				    << " Local " << lcoord.x() << ":" 
				    << lcoord.y() << " ID " << id_.iLay << ":"
				    << id_.iSec1 << ":" << id_.iSec2 << ":"
				    << id_.iCell1 << ":" << id_.iCell2 
				    << " Global " << glob;
#endif
    }
  } 
  return glob;
}

HGCalGeometry::CornersVec HGCalGeometry::getCorners(const DetId& id) const {

  unsigned int ncorner = ((m_det == DetId::HGCalHSc) ? FlatTrd::ncorner_ :
			  FlatHexagon::ncorner_);
  HGCalGeometry::CornersVec co (ncorner, GlobalPoint(0,0,0));
  unsigned int cellIndex =  indexFor(id);
  if ((cellIndex <  m_cellVec.size() && m_det != DetId::HGCalHSc) ||
      (cellIndex <  m_cellVec2.size() && m_det == DetId::HGCalHSc)) {
    HGCalTopology::DecodedDetId id_ = m_topology.decode(id);
    std::pair<float,float> xy;
    if ((mode_ == HGCalGeometryMode::Hexagon) ||
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      xy = m_topology.dddConstants().locateCellHex(id_.iCell1,id_.iSec1,true);
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      xy = m_topology.dddConstants().locateCellTrap(id_.iLay,id_.iSec1,
						    id_.iCell1,true);
    } else {
      xy = m_topology.dddConstants().locateCell(id_.iLay,id_.iSec1,id_.iSec2,
						id_.iCell1,id_.iCell2,true,false);
    }
    if (m_det == DetId::HGCalHSc) {
      float dx = 0.5*m_cellVec2[cellIndex].param()[11];
      float dz = m_cellVec2[cellIndex].param()[0];
      static const int signx[] = {-1,-1,1,1,-1,-1,1,1};
      static const int signy[] = {-1,1,1,-1,-1,1,1,-1};
      static const int signz[] = {-1,-1,-1,-1,1,1,1,1};
      for (unsigned int i = 0; i < ncorner; ++i) {
	const HepGeom::Point3D<float> lcoord(xy.first+signx[i]*dx,xy.second+signy[i]*dx,signz[i]*dz);
	co[i] = m_cellVec2[cellIndex].getPosition(lcoord);
      }
    } else {
      float dx = m_cellVec[cellIndex].param()[1];
      float dy = m_cellVec[cellIndex].param()[2];
      float dz = m_cellVec[cellIndex].param()[0];
      static const int signx[] = {0,-1,-1,0,1,1,0,-1,-1,0,1,1};
      static const int signy[] = {-2,-1,1,2,1,-1,-2,-1,1,2,1,-1};
      static const int signz[] = {-1,-1,-1,-1,-1,-1,1,1,1,1,1,1};
      for (unsigned int i = 0; i < ncorner; ++i) {
	const HepGeom::Point3D<float> lcoord(xy.first+signx[i]*dx,xy.second+signy[i]*dy,signz[i]*dz);
	co[i] = m_cellVec[cellIndex].getPosition(lcoord);
      }
    }
  }
  return co;
}

HGCalGeometry::CornersVec HGCalGeometry::get8Corners(const DetId& id) const {
  
  unsigned int ncorner = FlatTrd::ncorner_;
  HGCalGeometry::CornersVec co (ncorner, GlobalPoint(0,0,0));
  unsigned int cellIndex =  indexFor(id);
  if ((cellIndex <  m_cellVec.size() && m_det != DetId::HGCalHSc) ||
      (cellIndex <  m_cellVec2.size() && m_det == DetId::HGCalHSc)) {
    HGCalTopology::DecodedDetId id_ = m_topology.decode(id);
    std::pair<float,float> xy;
    if ((mode_ == HGCalGeometryMode::Hexagon) ||
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      xy = m_topology.dddConstants().locateCellHex(id_.iCell1,id_.iSec1,true);
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      xy = m_topology.dddConstants().locateCellTrap(id_.iLay,id_.iSec1,
						    id_.iCell1,true);
    } else {
      xy = m_topology.dddConstants().locateCell(id_.iLay,id_.iSec1,id_.iSec2,
						id_.iCell1,id_.iCell2,true,false);
    }
    float dx = ((m_det == DetId::HGCalHSc) ? 0.5*m_cellVec2[cellIndex].param()[11] :
		m_cellVec[cellIndex].param()[1]);
    float dz = ((m_det == DetId::HGCalHSc) ? m_cellVec2[cellIndex].param()[0] :
		m_cellVec[cellIndex].param()[0]);
    static const int signx[] = {-1,-1,1,1,-1,-1,1,1};
    static const int signy[] = {-1,1,1,-1,-1,1,1,-1};
    static const int signz[] = {-1,-1,-1,-1,1,1,1,1};
    for (unsigned int i = 0; i < ncorner; ++i) {
      const HepGeom::Point3D<float> lcoord(xy.first+signx[i]*dx,
					   xy.second+signy[i]*dx,signz[i]*dz);
      co[i] = ((m_det == DetId::HGCalHSc) ? 
	       (m_cellVec2[cellIndex].getPosition(lcoord)) :
	       (m_cellVec[cellIndex].getPosition(lcoord)));
    }
  }
  return co;
}

DetId HGCalGeometry::getClosestCell(const GlobalPoint& r) const {
  unsigned int cellIndex = getClosestCellIndex(r);
  if ((cellIndex <  m_cellVec.size() && m_det != DetId::HGCalHSc) ||
      (cellIndex <  m_cellVec2.size() && m_det == DetId::HGCalHSc)) {
    HGCalTopology::DecodedDetId id_ = m_topology.decode(m_validGeomIds[cellIndex]);
    HepGeom::Point3D<float> local;
    if (r.z() > 0) {
      local    = HepGeom::Point3D<float>(r.x(),r.y(),0);
      id_.zSide = 1;
    } else {
      local = HepGeom::Point3D<float>(-r.x(),r.y(),0);
      id_.zSide =-1;
    }
    if ((mode_ == HGCalGeometryMode::Hexagon) ||
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      const auto & kxy = 
	m_topology.dddConstants().assignCell(local.x(),local.y(),id_.iLay,
					     id_.iType,true);
      id_.iCell1  = kxy.second;
      id_.iSec1   = kxy.first;
      id_.iType   = m_topology.dddConstants().waferTypeT(kxy.first);
      if (id_.iType != 1) id_.iType = -1;
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      id_.iLay = m_topology.dddConstants().getLayer(r.z(),true);
      const auto & kxy = 
	m_topology.dddConstants().assignCellTrap(r.x(),r.y(),r.z(),
						 id_.iLay,true);
      id_.iSec1  = kxy[0];
      id_.iCell1 = kxy[1];
      id_.iType  = kxy[2];
    } else {
      id_.iLay = m_topology.dddConstants().getLayer(r.z(),true);
      const auto & kxy = 
	m_topology.dddConstants().assignCellHex(local.x(),local.y(),id_.iLay,
						true);
      id_.iSec1  = kxy[0]; id_.iSec2  = kxy[1]; id_.iType = kxy[2];
      id_.iCell1 = kxy[3]; id_.iCell2 = kxy[4];
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "getClosestCell: local " << local 
				  << " Id " << id_.zSide << ":" << id_.iLay 
				  << ":" << id_.iSec1 << ":" << id_.iSec2
				  << ":" << id_.iType << ":" << id_.iCell1
				  << ":" << id_.iCell2;
#endif

    //check if returned cell is valid
    if (id_.iCell1>=0) return m_topology.encode(id_);
  }

  //if not valid or out of bounds return a null DetId
  return DetId();
}

HGCalGeometry::DetIdSet HGCalGeometry::getCells(const GlobalPoint& r, double dR)  const {
   HGCalGeometry::DetIdSet dss;
   return dss;
}

std::string HGCalGeometry::cellElement() const {
  if      (m_subdet == HGCEE  || m_det == DetId::HGCalEE)  return "HGCalEE";
  else if (m_subdet == HGCHEF || m_det == DetId::HGCalHSi) return "HGCalHEFront";
  else if (m_subdet == HGCHEB || m_det == DetId::HGCalHSc) return "HGCalHEBack";
  else                                                     return "Unknown";
}

unsigned int HGCalGeometry::indexFor(const DetId& detId) const {
  unsigned int cellIndex = ((m_det == DetId::HGCalHSc) ? m_cellVec2.size() :
			    m_cellVec.size());
  if (detId != DetId()) {
    DetId geomId = getGeometryDetId(detId);
    cellIndex = m_topology.detId2denseGeomId(geomId);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "indexFor " << std::hex << detId.rawId() 
				  << ":" << geomId.rawId() << std::dec 
				  << " index " << cellIndex;
#endif
  }
  return cellIndex;
}

unsigned int HGCalGeometry::sizeForDenseIndex() const {
  return m_topology.totalGeomModules();
}

const CaloCellGeometry* HGCalGeometry::getGeometryRawPtr(uint32_t index) const {
  // Modify the RawPtr class
  if (m_det == DetId::HGCalHSc) {
    if (m_cellVec2.size() < index) return nullptr;
    const CaloCellGeometry* cell(&m_cellVec2[index]);
    return (nullptr == cell->param() ? nullptr : cell);
  } else {
    if (m_cellVec2.size() < index) return nullptr;
    const CaloCellGeometry* cell(&m_cellVec[index]);
    return (nullptr == cell->param() ? nullptr : cell);
  }
}

std::shared_ptr<const CaloCellGeometry> HGCalGeometry::cellGeomPtr(uint32_t index) const {
  if ((index >= m_cellVec.size()  && m_det != DetId::HGCalHSc) ||
      (index >= m_cellVec2.size() && m_det == DetId::HGCalHSc) ||
      (m_validGeomIds[index].rawId() == 0)) return nullptr;
  static const auto do_not_delete = [](const void*){};
  if (m_det == DetId::HGCalHSc) {
    auto cell = std::shared_ptr<const CaloCellGeometry>(&m_cellVec2[index],do_not_delete);
    if (nullptr == cell->param()) return nullptr;
    return cell;
  } else {
    auto cell = std::shared_ptr<const CaloCellGeometry>(&m_cellVec[index],do_not_delete);
    if (nullptr == cell->param()) return nullptr;
    return cell;
  }
}

std::shared_ptr<const CaloCellGeometry> HGCalGeometry::cellGeomPtr(uint32_t index, const GlobalPoint& pos) const {
  if ((index >= m_cellVec.size()  && m_det != DetId::HGCalHSc) ||
      (index >= m_cellVec2.size() && m_det == DetId::HGCalHSc) ||
      (m_validGeomIds[index].rawId() == 0)) return nullptr;
  if (pos == GlobalPoint()) return cellGeomPtr(index);
  if (m_det == DetId::HGCalHSc) {
    auto cell = std::make_shared<FlatTrd>(m_cellVec2[index]);
    cell->setPosition(pos);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "cellGeomPtr " << index << ":" << cell;
#endif
    if (nullptr == cell->param()) return nullptr;
    return cell;
  } else {
    auto cell = std::make_shared<FlatHexagon>(m_cellVec[index]);
    cell->setPosition(pos);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "cellGeomPtr " << index << ":" << cell;
#endif
    if (nullptr == cell->param()) return nullptr;
    return cell;
  }
}

void HGCalGeometry::addValidID(const DetId& id) {
  edm::LogError("HGCalGeom") << "HGCalGeometry::addValidID is not implemented";
}

unsigned int HGCalGeometry::getClosestCellIndex (const GlobalPoint& r) const {

  return ((m_det == DetId::HGCalHSc) ? getClosestCellIndex(r,m_cellVec2) : getClosestCellIndex(r,m_cellVec));
}

template<class T>
unsigned int HGCalGeometry::getClosestCellIndex (const GlobalPoint& r,
						 const std::vector<T>& vec) const {

  float phip = r.phi();
  float zp   = r.z();
  float dzmin(9999), dphimin(9999), dphi10(0.175);
  unsigned int cellIndex =  vec.size();
  for (unsigned int k=0; k<vec.size(); ++k) {
    float dphi = phip-vec[k].phiPos();
    while (dphi >   M_PI) dphi -= 2*M_PI;
    while (dphi <= -M_PI) dphi += 2*M_PI;
    if (std::abs(dphi) < dphi10) {
      float dz = std::abs(zp - vec[k].getPosition().z());
      if (dz < (dzmin+0.001)) {
	dzmin     = dz;
	if (std::abs(dphi) < (dphimin+0.01)) {
	  cellIndex = k;
	  dphimin   = std::abs(dphi);
	} else {
	  if (cellIndex >= vec.size()) cellIndex = k;
	}
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "getClosestCellIndex::Input " << zp << ":" 
				<< phip << " Index " << cellIndex;
  if (cellIndex < vec.size()) 
    edm::LogVerbatim("HGCalGeom") << " Cell z " 
				  << vec[cellIndex].getPosition().z() 
				  << ":" << dzmin << " phi " 
				  << vec[cellIndex].phiPos() << ":" << dphimin;
#endif
  return cellIndex;
}


// FIXME: Change sorting algorithm if needed
namespace {
  struct rawIdSort {
    bool operator()( const DetId& a, const DetId& b ) {
      return( a.rawId() < b.rawId());
    }
  };
}

 void HGCalGeometry::sortDetIds( void ) {
  m_validIds.shrink_to_fit();
  std::sort( m_validIds.begin(), m_validIds.end(), rawIdSort());
}

void HGCalGeometry::getSummary(CaloSubdetectorGeometry::TrVec&  trVector,
			       CaloSubdetectorGeometry::IVec&   iVector,
			       CaloSubdetectorGeometry::DimVec& dimVector,
			       CaloSubdetectorGeometry::IVec& dinsVector ) const {

  unsigned int numberOfCells = m_topology.totalGeomModules(); // total Geom Modules both sides
  unsigned int numberOfShapes = k_NumberOfShapes;
  unsigned int numberOfParametersPerShape = 
    ((m_det == DetId::HGCalHSc) ? (unsigned int)(k_NumberOfParametersPerTrd) :
     (unsigned int)(k_NumberOfParametersPerHex));

  trVector.reserve( numberOfCells * numberOfTransformParms());
  iVector.reserve( numberOfCells );
  dimVector.reserve( numberOfShapes * numberOfParametersPerShape );
  dinsVector.reserve( numberOfCells );
  
  for (unsigned itr=0; itr<m_topology.dddConstants().getTrFormN(); ++itr) {
    HGCalParameters::hgtrform mytr = m_topology.dddConstants().getTrForm(itr);
    int layer  = mytr.lay;

    if ((mode_ == HGCalGeometryMode::Hexagon) ||
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      for (int wafer=0; wafer<m_topology.dddConstants().sectors(); ++wafer) {
        if (m_topology.dddConstants().waferInLayer(wafer,layer,true)) {
	  HGCalParameters::hgtrap vol = m_topology.dddConstants().getModule(wafer, true, true);
	  ParmVec params( numberOfParametersPerShape, 0 );
	  params[0] = vol.dz;
	  params[1] = vol.cellSize;
	  params[2] = twoBysqrt3_*params[1];
	  dimVector.insert( dimVector.end(), params.begin(), params.end());
	}
      }
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      int indx = m_topology.dddConstants().layerIndex(layer,true);
      for (int md=m_topology.dddConstants().getParameter()->firstModule_[indx];
	   md<=m_topology.dddConstants().getParameter()->lastModule_[indx];
	   ++md) {
	HGCalParameters::hgtrap vol = m_topology.dddConstants().getModule(md, true, true);
	ParmVec params( numberOfParametersPerShape, 0 );
	params[1] = params[2] = 0;
	params[3] = params[7] = vol.h;
	params[4] = params[8] = vol.bl;
	params[5] = params[9] = vol.tl;
	params[6] = params[10]= vol.alpha;
	params[11]= vol.cellSize;
	dimVector.insert( dimVector.end(), params.begin(), params.end());
      }
    } else {
      for (int wafer=0; wafer<m_topology.dddConstants().sectors(); ++wafer) {
        if (m_topology.dddConstants().waferInLayer(wafer,layer,true)) {
	  HGCalParameters::hgtrap vol = m_topology.dddConstants().getModule(wafer, true, true);
	  ParmVec params( numberOfParametersPerShape, 0 );
	  params[0] = vol.dz;
	  params[1] = vol.cellSize;
	  params[2] = twoBysqrt3_*params[1];
	  dimVector.insert( dimVector.end(), params.begin(), params.end());
	}
      }
    }
  }
  
  for (unsigned int i( 0 ); i < numberOfCells; ++i) {
    DetId detId = m_validGeomIds[i];
    int layer(0);
    if ((mode_ == HGCalGeometryMode::Hexagon) ||
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      layer     = HGCalDetId(detId).layer();
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      layer     = HGCScintillatorDetId(detId).layer();
    } else if (m_topology.isHFNose()) {
      layer     = HFNoseDetId(detId).layer();
    } else {
      layer     = HGCSiliconDetId(detId).layer();
    }
    dinsVector.emplace_back(m_topology.detId2denseGeomId( detId ));
    iVector.emplace_back( layer );
    
    Tr3D tr;
    auto ptr =  cellGeomPtr( i );
    if ( nullptr != ptr ) {
      ptr->getTransform( tr, ( Pt3DVec* ) nullptr );

      if( Tr3D() == tr ) { // there is no rotation
	const GlobalPoint& gp( ptr->getPosition()); 
	tr = HepGeom::Translate3D( gp.x(), gp.y(), gp.z());
      }

      const CLHEP::Hep3Vector tt( tr.getTranslation());
      trVector.emplace_back( tt.x());
      trVector.emplace_back( tt.y());
      trVector.emplace_back( tt.z());
      if (6 == numberOfTransformParms()) {
	const CLHEP::HepRotation rr( tr.getRotation());
	const ROOT::Math::Transform3D rtr( rr.xx(), rr.xy(), rr.xz(), tt.x(),
					   rr.yx(), rr.yy(), rr.yz(), tt.y(),
					   rr.zx(), rr.zy(), rr.zz(), tt.z());
	ROOT::Math::EulerAngles ea;
	rtr.GetRotation( ea );
	trVector.emplace_back( ea.Phi());
	trVector.emplace_back( ea.Theta());
	trVector.emplace_back( ea.Psi());
      }
    }
  }
}

DetId HGCalGeometry::getGeometryDetId(DetId detId) const {
  DetId geomId;
  if ((mode_ == HGCalGeometryMode::Hexagon) || 
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    geomId = static_cast<DetId>(HGCalDetId(detId).geometryCell());
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    geomId = static_cast<DetId>(HGCScintillatorDetId(detId).geometryCell());
  } else if (m_topology.isHFNose()) {
    geomId = static_cast<DetId>(HFNoseDetId(detId).geometryCell());
  } else {
    geomId = static_cast<DetId>(HGCSiliconDetId(detId).geometryCell());
  }
  return geomId;
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalGeometry);
