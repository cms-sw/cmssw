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

   assert( gid.isEK() ) ;
   int iq = topology().dddConstants().quadrant (EKDetId(id).ix(), EKDetId(id).iy());
   unsigned int result = 0;
   if (iq == 2 || iq == 3) result += 1; //positive X
   if (EKDetId(id).zside() > 0) result += 2;
   return result;
}

DetId 
EcalShashlikGeometry::detIdFromLocalAlignmentIndex( unsigned int iLoc ) const
{
  int nCols = topology().dddConstants().getCols ();
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
}


int EcalShashlikGeometry::xindex( CCGFloat x,
				     CCGFloat z ) const
{
  int iz = (z>0)?1:0;
  double xRef = x / fabs (mSide[iz].zMean); 
  int result = int (0.5 + mSide[iz].ixMin +
		    (xRef - mSide[iz].xMin) / (mSide[iz].xMax - mSide[iz].xMin) * 
		    (mSide[iz].ixMax - mSide[iz].ixMin));
  return result;
}

int EcalShashlikGeometry::yindex( CCGFloat y,
				     CCGFloat z ) const
{
  int iz = (z>0)?1:0;
  double yRef = y / fabs (mSide[iz].zMean); 
  int result = int (0.5 + mSide[iz].iyMin +
		    (yRef - mSide[iz].yMin) / (mSide[iz].yMax - mSide[iz].yMin) * 
		    (mSide[iz].iyMax - mSide[iz].iyMin));
  return result;
}

EKDetId 
EcalShashlikGeometry::gId( float x, 
			   float y, 
			   float z ) const
{
  EKDetId result;
  int ix0 = xindex (x, z);
  int iy0 = xindex (y, z);
  int iz = z>0?1:0;
  int zSide = z>0?1:-1;
  if (ix0 >= mSide[iz].ixMin && ix0 <= mSide[iz].ixMax && 
      iy0 >=mSide[iz].ixMin && iy0 <= mSide[iz].iyMax &&
      topology().valid ((result = EKDetId (ix0, iy0, 0, 0, zSide)).rawId())) { 
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
  std::cerr << "EcalShashlikGeometry::getClosestCell-> Not implemented..." << std::endl;
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

void
EcalShashlikGeometry::newCell( const GlobalPoint& f1 ,
			     const GlobalPoint& f2 ,
			     const GlobalPoint& f3 ,
			     const CCGFloat*    parm ,
			     const DetId&       detId   ) 
{
  const uint32_t cellIndex (topology().cell2denseId(detId));
  if (cellIndex >= m_cellVec.size ()) m_cellVec.resize (cellIndex+1);
  m_cellVec[ cellIndex ] = TruncatedPyramid( cornersMgr(), f1, f2, f3, parm ) ;
  m_validIds.push_back( detId ) ;
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
   return ( m_cellVec.size() < index ||
	    0 == cell->param() ? 0 : cell ) ;
}

const ShashlikTopology& EcalShashlikGeometry::topology () const {
  if (initializedTopology) return mTopology;
  throw cms::Exception("Topology") << "EcalShashlikGeometry: attempt to access uninitialized ECAL Shashlik topology";
}
