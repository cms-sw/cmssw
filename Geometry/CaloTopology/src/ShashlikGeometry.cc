/*
 * Geometry for Shashlik ECAL
 * This geometry is essentially driven by topology, 
 * which is thus encapsulated in this class. 
 * This makes this geometry not suitable to be loaded
 * by regular CaloGeometryLoader<T>
 * Fedor Ratnikov, Apr. 8 2014
 */
#include "Geometry/CaloTopology/interface/ShashlikGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "FWCore/Utilities/interface/Exception.h"

ShashlikGeometry::ShashlikGeometry()
{
}

ShashlikGeometry::ShashlikGeometry(const ShashlikTopology& topology)
  : mTopology (topology)
{
}

ShashlikGeometry::~ShashlikGeometry() 
{
}

void ShashlikGeometry::fillNamedParams (DDFilteredView fv) {}

void 
ShashlikGeometry::initializeParms() // assume only m_cellVec are available
{
  size_t nZ[2] = {0, 0};
  for (size_t i=0; i < m_cellVec.size(); ++i)
  {
    const CaloCellGeometry* cell ( cellGeomPtr(i) ) ;
    if (cell) {
      size_t iz = cell->getPosition().z() > 0 ? 1 : 0;
      ++nZ[iz];
      mSide[iz].zMean += cell->getPosition().z();
      double xRef = cell->getPosition().x() / fabs (cell->getPosition().z());
      double yRef = cell->getPosition().y() / fabs (cell->getPosition().z());
      EKDetId myId (topology().denseId2cell(i));
      int ix ( myId.ix() ) ;
      int iy ( myId.iy() ) ;
      if (xRef > mSide[iz].xMax) {
	mSide[iz].xMax = xRef;
	mSide[iz].ixMax = ix;
      }
      if (xRef < mSide[iz].xMin) {
	mSide[iz].xMin = xRef;
	mSide[iz].ixMin = ix;
      }
      if (yRef > mSide[iz].yMax) {
	mSide[iz].yMax = yRef;
	mSide[iz].iyMax = iy;
      }
      if (yRef < mSide[iz].yMin) {
	mSide[iz].yMin = yRef;
	mSide[iz].iyMin = iy;
	}
    }
  }
  for (size_t iz = 0; iz < 2; ++iz) {
    if (nZ[iz]>0) {
      mSide[iz].zMean /= nZ[iz];
      double absZ = fabs (mSide[iz].zMean);
      mSide[iz].xMin *= absZ;
      mSide[iz].xMax *= absZ;
      mSide[iz].yMin *= absZ;
      mSide[iz].yMax *= absZ;
    } 
  }

  // cross check
  for (size_t i = 0; i < m_validIds.size(); ++i) {
    if (i != mTopology.cell2denseId(m_validIds[i])) {
      edm::LogError("HGCalGeom") << "ShashlikGeometry::initializeParms-> inconsistent geometry structure " 
				 << EKDetId(m_validIds[i]) << " -> " << i << " -> " << EKDetId(mTopology.cell2denseId(m_validIds[i]));	
    }
  }
}

void ShashlikGeometry::addValidID(const DetId& id) {
  edm::LogError("HGCalGeom") << "ShashlikGeometry::addValidID is not implemented";
}

void
ShashlikGeometry::newCell( const GlobalPoint& f1 ,
			     const GlobalPoint& f2 ,
			     const GlobalPoint& f3 ,
			     const CCGFloat*    parm ,
			     const DetId&       detId   ) 
{
  const uint32_t cellIndex (topology().cell2denseId(detId));
  if (cellIndex >= m_cellVec.size ()) m_cellVec.resize (cellIndex+1);
  if (cellIndex >= m_validIds.size ()) m_validIds.resize (cellIndex+1);
  m_cellVec[ cellIndex ] = TruncatedPyramid( cornersMgr(), f1, f2, f3, parm ) ;
  m_validIds[ cellIndex ] = detId ;
//   std::cout << "ShashlikGeometry::newCell-> front:" << f1.x() << '/' << f1.y() << '/' << f1.z() 
// 	    << " back:" <<  f2.x() << '/' << f2.y() << '/' << f2.z()
// 	    << " id:" << EKDetId (detId)
// 	    << std::endl; 
}

const CaloCellGeometry* ShashlikGeometry::getGeometry( const DetId& id ) const {
  if (id == DetId()) return 0; // nothing to get
  EKDetId geoId(id);
  const uint32_t cellIndex (topology().cell2denseId(geoId));
  const CaloCellGeometry* result = cellGeomPtr (cellIndex);
  if (m_validIds[cellIndex] != geoId.rawId()) {
    edm::LogError("HGCalGeom") << "ShashlikGeometry::getGeometry-> inconsistent geometry structure " 
			       << id.rawId() << "(" << cellIndex << "->" << topology().denseId2cell(cellIndex).rawId() << ")" 
			       << " inside: " << EKDetId(m_validIds[cellIndex]) << "(" <<  m_validIds[cellIndex].rawId() << "->" << topology().cell2denseId(m_validIds[cellIndex]) << ")"
			       << " given: " << geoId << "(" <<  geoId.rawId() << ")"
			       << " ref " << EKDetId (topology().denseId2cell(cellIndex));
    return 0;
  }
  return result;
}



const CaloCellGeometry* 
ShashlikGeometry::cellGeomPtr( uint32_t index ) const
{
   const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
   if ((index >= m_cellVec.size()) || (0 == cell->param()) || (m_validIds[index].rawId() == 0)) return 0;
   return cell;
}

const ShashlikTopology& ShashlikGeometry::topology () const {
  return mTopology;
}

