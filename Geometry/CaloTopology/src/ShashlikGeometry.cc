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
  : initializedTopology (false)
{
}

ShashlikGeometry::ShashlikGeometry(const ShashlikTopology& topology)
  : mTopology (topology),
    initializedTopology (true)
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
  if (!initializedTopology) {
    // recover topology
    int ixMiddle = (mSide[1].ixMin + mSide[1].ixMax -1) / 2;
    int iyMiddle = (mSide[1].iyMin + mSide[1].iyMax -1) / 2;
    const int BIG_ROW = 25;
    std::vector<int> firstY (BIG_ROW, 999);
    std::vector<int> lastY (BIG_ROW, 0);
    for (size_t i=0; i < m_cellVec.size(); ++i) {
      const CaloCellGeometry* cell ( cellGeomPtr(i) ) ;
      if (cell) {
	if (cell->getPosition().z() > 0 && 
	    cell->getPosition().x() > 0 && 
	    cell->getPosition().y()> 0) {
	  int ix = xindex (cell->getPosition().x(), 
			   cell->getPosition().z()) - ixMiddle;
	  int iy = yindex (cell->getPosition().x(), 
			   cell->getPosition().z()) - iyMiddle + 1;
	  if (ix < 0 || ix >= BIG_ROW || iy < 1 || iy > BIG_ROW) {
	    throw cms::Exception("Topology") << "ShashlikGeometry: strange cells in input Geometry ix/iy: " << ix << '/' << iy;
	  }
	  if (iy < firstY[ix]) firstY[ix] = iy; 
	  if (iy > lastY[ix]) lastY[ix] = iy; 
	}
      }
    }
    // check consistency
    bool active = true;
    int nRow = 0;
    for (int i = 0; i < BIG_ROW; ++i) {
      if (active) {
	if (lastY[i] >= 1 && firstY[i] <= lastY[i]) {
	  continue; // OK
	}
	else {
	  active = false;
	  nRow = i;
	} 
      }
      if (!active) {
	if (lastY[i] != 0 || firstY[i] != 999) {
	  throw cms::Exception("Topology") << "ShashlikGeometry: inconsistent initialization: ix/minY/maxY: " << i << '/' << firstY[i] << '/' << lastY[i];
	}
      }
    }
    firstY.resize (nRow);
    lastY.resize (nRow);
    ShashlikDDDConstants dddConstants;
    dddConstants.loadSpecPars (firstY, lastY);
    mTopology = ShashlikTopology (dddConstants);
    initializedTopology = true;
  }
  // cross check
  for (size_t i = 0; i < m_validIds.size(); ++i) {
    if (i != mTopology.cell2denseId(m_validIds[i])) {
      edm::LogError("HGCalGeom") << "ShashlikGeometry::initializeParms-> inconsistent geometry structure " 
				 << EKDetId(m_validIds[i]) << " -> " << i << " -> " << EKDetId(mTopology.cell2denseId(m_validIds[i]));	
    }
  }
}


int ShashlikGeometry::xindex( CCGFloat x,
			      CCGFloat z ) const
{
  int iz = (z>0)?1:0;
  double xRef = x * fabs (mSide[iz].zMean / z); 
  int result = int (0.5 + mSide[iz].ixMin +
		    (xRef - mSide[iz].xMin) / (mSide[iz].xMax - mSide[iz].xMin) * 
		    (mSide[iz].ixMax - mSide[iz].ixMin));
//   std::cout << "ShashlikGeometry::xindex-> "
// 	    << "min/max/ref:" << mSide[iz].xMin << '/' << mSide[iz].xMax << '/' << xRef
// 	    << " imin/max:" << mSide[iz].ixMin << '/' << mSide[iz].ixMax
// 	    << " result: " << result
// 	    << std::endl;
  return result;
}

int ShashlikGeometry::yindex( CCGFloat y,
				     CCGFloat z ) const
{
  int iz = (z>0)?1:0;
  double yRef = y * fabs (mSide[iz].zMean / z); 
  int result = int (0.5 + mSide[iz].iyMin +
		    (yRef - mSide[iz].yMin) / (mSide[iz].yMax - mSide[iz].yMin) * 
		    (mSide[iz].iyMax - mSide[iz].iyMin));
//   std::cout << "ShashlikGeometry::yindex-> "
// 	    << "min/max/ref:" << mSide[iz].yMin << '/' << mSide[iz].yMax << '/' << yRef
// 	    << " imin/max:" << mSide[iz].iyMin << '/' << mSide[iz].iyMax
// 	    << " z/zref:" << z << '/' << mSide[iz].zMean
// 	    << " result: " << result
// 	    << std::endl;
  return result;
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
  EKDetId geoId = EKDetId(id).geometryCell ();
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
  if (initializedTopology) return mTopology;
  throw cms::Exception("Topology") << "ShashlikGeometry: attempt to access uninitialized ECAL Shashlik topology";
}

