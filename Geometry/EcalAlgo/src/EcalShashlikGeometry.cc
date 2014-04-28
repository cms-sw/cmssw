#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/EcalAlgo/interface/EcalShashlikGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
typedef HepGeom::Plane3D<CCGFloat> Pl3D     ;

namespace {
  double combined (const CaloCellGeometry::Pt3D& v1, const CaloCellGeometry::Pt3D& v2, const CaloCellGeometry::Pt3D& v3) {
    return v1.x()*v2.y()*v3.z()+
      v1.y()*v2.z()*v3.x() + 
      v1.z()*v2.x()*v3.y() -
      v1.x()*v2.z()*v3.y() -
      v1.y()*v2.x()*v3.z() -
      v1.z()*v2.y()*v3.x();
  }

  bool sameSide (const GlobalPoint& point, const CaloCellGeometry::CornersVec& corners, int iref, int i1, int i2, int icalib) {
    CaloCellGeometry::Pt3D v1 (corners[i1].x()-corners[iref].x(), corners[i1].y()-corners[iref].y(), corners[i1].z()-corners[iref].z()) ;
    CaloCellGeometry::Pt3D v2 (corners[i2].x()-corners[iref].x(), corners[i2].y()-corners[iref].y(), corners[i2].z()-corners[iref].z()) ;
    CaloCellGeometry::Pt3D vpoint (point.x()-corners[iref].x(), point.y()-corners[iref].y(), point.z()-corners[iref].z()) ;
    CaloCellGeometry::Pt3D vcalib (corners[icalib].x()-corners[iref].x(), corners[icalib].y()-corners[iref].y(), corners[icalib].z()-corners[iref].z()) ;
//     std::cout << "sameSide-> " << point << " ref/1/2/calib:" 
// 	      << corners[iref] << " : "
// 	      << corners[i1] << " : "
// 	      << corners[i2] << " : "
// 	      << corners[icalib] << "  "
// 	      << " result: " << bool (combined (v1, v2, vpoint) * combined (v1, v2, vcalib) >= 0)
// 	      << std::endl;
    return combined (v1, v2, vpoint) * combined (v1, v2, vcalib) >= 0; 
  }  


  int pointInCell (const GlobalPoint& r, const EcalShashlikGeometry& geometry, EKDetId cell) {
//     std::cout << "pointInCell-> " << r << " cell: " << cell << std::endl;
    const CaloCellGeometry* geom = geometry.getGeometry (cell);
    if (!geom) return -1;
    const CaloCellGeometry::CornersVec& corners =  geom->getCorners();
    if (!sameSide (r, corners, 0, 1, 3, 4)) return 1;  // -z
    if (!sameSide (r, corners, 6, 5, 7, 2)) return 2;  // +z
    if (!sameSide (r, corners, 0, 3, 4, 1)) return 3;  // -x
    if (!sameSide (r, corners, 6, 2, 5, 7)) return 4;  // +x
    if (!sameSide (r, corners, 6, 7, 2, 5)) return 6;  // -y
    if (!sameSide (r, corners, 0, 4, 1, 3)) return 5;  // +y
    return 0;
  }
};

EcalShashlikGeometry::EcalShashlikGeometry()
  : initializedTopology (false)
{
}

EcalShashlikGeometry::EcalShashlikGeometry(const ShashlikTopology& topology)
  : mTopology (topology),
    initializedTopology (true)
{
}

EcalShashlikGeometry::~EcalShashlikGeometry() 
{
}

void EcalShashlikGeometry::allocateCorners (size_t n) {
  CaloSubdetectorGeometry::allocateCorners (n);
  m_cellVec.reserve (n);
}

unsigned int
EcalShashlikGeometry::alignmentTransformIndexLocal( const DetId& id ) const
{
   const CaloGenericDetId gid ( id ) ;

   assert( CaloGenericDetId(id).isEK() ) ;
   unsigned int result = topology().dddConstants().positiveX (EKDetId(id).ix()) ? 1 : 0;
   if (EKDetId(id).zside() > 0) result += 2;
   return result;
}

DetId 
EcalShashlikGeometry::detIdFromLocalAlignmentIndex( unsigned int iLoc ) const
{
  int nCols = topology().dddConstants().getModuleCols ();
  int ix = iLoc%2 ? nCols*3/4 : nCols/4;
  int iy = nCols/2;
  int iz = iLoc >= 2 ? 1 : -1;
  return EKDetId (ix, iy, 0, 0, iz);
}

unsigned int
EcalShashlikGeometry::alignmentTransformIndexGlobal( const DetId& /*id*/ )
{
   return (unsigned int)DetId::Ecal - 1 ;
}

void EcalShashlikGeometry::fillNamedParams (DDFilteredView fv) {
}

void 
EcalShashlikGeometry::initializeParms() // assume only m_cellVec are available
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
	    throw cms::Exception("Topology") << "EcalShashlikGeometry: strange cells in input Geometry ix/iy: " << ix << '/' << iy;
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
	  throw cms::Exception("Topology") << "EcalShashlikGeometry: inconsistent initialization: ix/minY/maxY: " << i << '/' << firstY[i] << '/' << lastY[i];
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
      edm::LogError("HGCalGeom") << "EcalShashlikGeometry::initializeParms-> inconsistent geometry structure " 
				 << EKDetId(m_validIds[i]) << " -> " << i << " -> " << EKDetId(mTopology.cell2denseId(m_validIds[i]));	
    }
  }
}


int EcalShashlikGeometry::xindex( CCGFloat x,
				     CCGFloat z ) const
{
  int iz = (z>0)?1:0;
  double xRef = x * fabs (mSide[iz].zMean / z); 
  int result = int (0.5 + mSide[iz].ixMin +
		    (xRef - mSide[iz].xMin) / (mSide[iz].xMax - mSide[iz].xMin) * 
		    (mSide[iz].ixMax - mSide[iz].ixMin));
//   std::cout << "EcalShashlikGeometry::xindex-> "
// 	    << "min/max/ref:" << mSide[iz].xMin << '/' << mSide[iz].xMax << '/' << xRef
// 	    << " imin/max:" << mSide[iz].ixMin << '/' << mSide[iz].ixMax
// 	    << " result: " << result
// 	    << std::endl;
  return result;
}

int EcalShashlikGeometry::yindex( CCGFloat y,
				     CCGFloat z ) const
{
  int iz = (z>0)?1:0;
  double yRef = y * fabs (mSide[iz].zMean / z); 
  int result = int (0.5 + mSide[iz].iyMin +
		    (yRef - mSide[iz].yMin) / (mSide[iz].yMax - mSide[iz].yMin) * 
		    (mSide[iz].iyMax - mSide[iz].iyMin));
//   std::cout << "EcalShashlikGeometry::yindex-> "
// 	    << "min/max/ref:" << mSide[iz].yMin << '/' << mSide[iz].yMax << '/' << yRef
// 	    << " imin/max:" << mSide[iz].iyMin << '/' << mSide[iz].iyMax
// 	    << " z/zref:" << z << '/' << mSide[iz].zMean
// 	    << " result: " << result
// 	    << std::endl;
  return result;
}

EKDetId 
EcalShashlikGeometry::gId( float x, 
			   float y, 
			   float z ) const
{
  EKDetId result;
  int ix0 = xindex (x, z);
  int iy0 = yindex (y, z);
  int iz = z>0?1:0;
  int zSide = z>0?1:-1;
//   std::cout << "EcalShashlikGeometry::gId-> x/y/z: " 
// 	    << x << '/' << y << '/' << z
// 	    << " indeces:" << ix0 << '/' << iy0 
// 	    << std::endl;
//   std::cout << "EcalShashlikGeometry::gId-> 1 " << EKDetId(result) 
// 	    << " valid:" << topology().valid ((result = EKDetId (ix0, iy0, 0, 0, zSide)).rawId()) << std::endl;
  if (ix0 >= std::min (mSide[iz].ixMin, mSide[iz].ixMax) && ix0 <= std::max (mSide[iz].ixMin, mSide[iz].ixMax) &&
      iy0 >= std::min (mSide[iz].iyMin, mSide[iz].iyMax) && iy0 <= std::max (mSide[iz].iyMin, mSide[iz].iyMax) &&
      topology().valid ((result = EKDetId (ix0, iy0, 0, 0, zSide)).rawId())) {
//     std::cout << "EcalShashlikGeometry::gId-> 1 " << EKDetId(result) << std::endl;
    return result; // first try is on target
  }
  // try nearby coordinates, spiraling out from center
  for (int i = 1; i < 6; ++i) {
    for(int k = 0; k < 8; ++k) {
      int ix = ix0 + (k==0 || k==4 || k==5) ? i : (k==1 || k>5) ? -i : 0;
      if (ix < mSide[iz].ixMin || ix0 > mSide[iz].ixMax) continue; 
      int iy = iy0 + (k==2 || k==4 || k==6) ? i : (k==3 || k==5 || k==7) ? -i : 0;
      if (iy < mSide[iz].iyMin || iy0 > mSide[iz].iyMax) continue; 
      if (topology().valid ((result = EKDetId (ix, iy, 0, 0, zSide)).rawId())) {
// 	std::cout << "EcalShashlikGeometry::gId-> 2 " << EKDetId(result) << std::endl;
	return result;
      }
    }
  }
  return EKDetId() ; // nowhere near any crystal
}


// Get closest cell, etc...
DetId 
EcalShashlikGeometry::getClosestCell( const GlobalPoint& r ) const 
{
  int nIterations = 3; // total iterative attempts
  DetId cellId = gId (r.x(), r.y(), r.z() ); // educated guess
  while (--nIterations >= 0) {
    if (cellId == DetId()) break; //invalid cell, stop
    int offset = pointInCell (r, *this, EKDetId(cellId));
//     std::cout << "EcalShashlikGeometry::getClosestCell-> iter " << nIterations 
// 	      << " point " << r 
// 	      << "cell:" << EKDetId(cellId) 
// 	      << " inside: " << offset <<std::endl;
    if (offset >= 0 && offset <= 2) return cellId; // disregard Z matching
    else if (offset == 3) cellId = mTopology.goWest (cellId);
    else if (offset == 4) cellId = mTopology.goEast (cellId);
    else if (offset == 5) cellId = mTopology.goNorth (cellId);
    else if (offset == 6) cellId = mTopology.goSouth (cellId);
  }
  return DetId();
}

CaloSubdetectorGeometry::DetIdSet 
EcalShashlikGeometry::getCells( const GlobalPoint& r, 
			      double             dR ) const 
{
  std::cerr << "EcalShashlikGeometry::getCells-> Not implemented..." << std::endl;
  return CaloSubdetectorGeometry::DetIdSet();
}

const EcalShashlikGeometry::OrderedListOfEBDetId*
EcalShashlikGeometry::getClosestBarrelCells( EKDetId id ) const
{
  std::cerr << "EcalShashlikGeometry::getClosestBarrelCells-> Not implemented..." << std::endl;
  return 0;
}

void
EcalShashlikGeometry::localCorners( Pt3DVec&        lc  ,
				  const CCGFloat* pv  ,
				  unsigned int   /*i*/,
				  Pt3D&           ref   )
{
   TruncatedPyramid::localCorners( lc, pv, ref ) ;
}

void EcalShashlikGeometry::addValidID(const DetId& id) {
  edm::LogError("HGCalGeom") << "EcalShashlikGeometry::addValidID is not implemented";
}

void
EcalShashlikGeometry::newCell( const GlobalPoint& f1 ,
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
//   std::cout << "EcalShashlikGeometry::newCell-> front:" << f1.x() << '/' << f1.y() << '/' << f1.z() 
// 	    << " back:" <<  f2.x() << '/' << f2.y() << '/' << f2.z()
// 	    << " id:" << EKDetId (detId)
// 	    << std::endl; 
}

const CaloCellGeometry* EcalShashlikGeometry::getGeometry( const DetId& id ) const {
  if (id == DetId()) return 0; // nothing to get
  EKDetId geoId = EKDetId(id).geometryCell ();
  const uint32_t cellIndex (topology().cell2denseId(geoId));
  const CaloCellGeometry* result = cellGeomPtr (cellIndex);
  if (m_validIds[cellIndex] != geoId.rawId()) {
    edm::LogError("HGCalGeom") << "EcalShashlikGeometry::getGeometry-> inconsistent geometry structure " 
			       << id.rawId() << "(" << cellIndex << "->" << topology().denseId2cell(cellIndex).rawId() << ")" 
			       << " inside: " << EKDetId(m_validIds[cellIndex]) << "(" <<  m_validIds[cellIndex].rawId() << "->" << topology().cell2denseId(m_validIds[cellIndex]) << ")"
			       << " given: " << geoId << "(" <<  geoId.rawId() << ")"
			       << " ref " << EKDetId (topology().denseId2cell(cellIndex));
    return 0;
  }
  return result;
}



CCGFloat 
EcalShashlikGeometry::avgAbsZFrontFaceCenter() const
{
  std::cerr << "EcalShashlikGeometry::avgAbsZFrontFaceCenter-> Not implemented..." << std::endl;
  return 0;
}

const CaloCellGeometry* 
EcalShashlikGeometry::cellGeomPtr( uint32_t index ) const
{
   const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
   if ((index >= m_cellVec.size()) || (0 == cell->param()) || (m_validIds[index].rawId() == 0)) return 0;
   return cell;
}

const ShashlikTopology& EcalShashlikGeometry::topology () const {
  if (initializedTopology) return mTopology;
  throw cms::Exception("Topology") << "EcalShashlikGeometry: attempt to access uninitialized ECAL Shashlik topology";
}

