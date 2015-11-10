/*
 * Geometry for High Granularity Calorimeter
 * This geometry is essentially driven by topology, 
 * which is thus encapsulated in this class. 
 * This makes this geometry not suitable to be loaded
 * by regular CaloGeometryLoader<T>
 */
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cmath>

#include <Math/Transform3D.h>
#include <Math/EulerAngles.h>

typedef CaloCellGeometry::Tr3D     Tr3D     ;

//#define DebugLog

HGCalGeometry::HGCalGeometry( const HGCalTopology& topology_ )
  : m_topology( topology_ ),
    m_cellVec( topology_.totalGeomModules()),
    m_validGeomIds( topology_.totalGeomModules()),
    m_halfType( topology_.detectorType()),
    m_subdet( topology_.subDetector())
{
  m_validIds.reserve( topology().totalModules());
#ifdef DebugLog
  std::cout << "Expected total # of Geometry Modules " 
	    << topology().totalGeomModules() << std::endl;
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
  TruncatedPyramid::localCorners( lc, pv, ref ) ;
}

void HGCalGeometry::newCell( const GlobalPoint& f1 ,
			     const GlobalPoint& f2 ,
			     const GlobalPoint& f3 ,
			     const CCGFloat*    parm ,
			     const DetId&       detId ) {
  const uint32_t cellIndex (topology().detId2denseGeomId(detId));
  DetId geomId = (detId.subdetId() == HGCEE ? 
		  (DetId)(HGCEEDetId(detId).geometryCell()) :
		  (DetId)(HGCHEDetId(detId).geometryCell()));
  m_cellVec.at( cellIndex ) = FlatTrd( cornersMgr(), f1, f2, f3, parm ) ;
  m_validGeomIds.at( cellIndex ) = geomId ;

  HGCalTopology::DecodedDetId id = topology().decode(detId);
  int cells = topology().dddConstants().maxCells(id.iLay,true);
  for (int cell = 0; cell < cells; ++cell) {
    id.iCell = cell;
    m_validIds.push_back( topology().encode(id));
    if (!m_halfType) {
      id.iSubSec = -id.iSubSec;
      m_validIds.push_back( topology().encode(id));
      id.iSubSec = -id.iSubSec;
    }
  }

#ifdef DebugLog
  std::cout << "HGCalGeometry::newCell-> [" << cellIndex << "]"
	    << " front:" << f1.x() << '/' << f1.y() << '/' << f1.z() 
     	    << " back:" <<  f2.x() << '/' << f2.y() << '/' << f2.z()
	    << " eta|phi " << m_cellVec[cellIndex].etaPos() << ":"
	    << m_cellVec[cellIndex].phiPos() << " id:";
  if (m_subdet == HGCEE) std::cout << HGCEEDetId(detId);
  else                   std::cout << HGCHEDetId(detId);
  std::cout << " with valid DetId from " << nOld << " to " << nNew
 	    << std::endl; 
  std::cout << "Cell[" << cellIndex << "] " << std::hex << geomId.rawId() 
	    << ":"  << m_validGeomIds[cellIndex].rawId() << std::dec << " "
	    << m_cellVec[cellIndex];
#endif
}

const CaloCellGeometry* HGCalGeometry::getGeometry(const DetId& id) const {
  if (id == DetId()) return 0; // nothing to get
  DetId geoId = (id.subdetId() == HGCEE ? 
		 (DetId)(HGCEEDetId(id).geometryCell()) : 
		 (DetId)(HGCHEDetId(id).geometryCell()));
  const uint32_t cellIndex (topology().detId2denseGeomId(geoId));
  /*
  if (cellIndex <  m_cellVec.size()) {
    HGCalTopology::DecodedDetId id_ = topology().decode(id);
    std::pair<float,float> xy = topology().dddConstants().locateCell(id_iCell,id_iLay,id_.iSubSec,true);
    const HepGeom::Point3D<float> lcoord(xy.first,xy.second,0);
    std::auto_ptr<FlatTrd> cellGeom(new FlatTrd(m_cellVec[cellIndex],lcoord));
    return cellGeom.release();
  }
  */
  return cellGeomPtr (cellIndex);

}

GlobalPoint HGCalGeometry::getPosition( const DetId& id ) const {

  unsigned int cellIndex =  indexFor(id);
  if (cellIndex <  m_cellVec.size()) {
    HGCalTopology::DecodedDetId id_ = topology().decode(id);
    std::pair<float,float> xy = topology().dddConstants().locateCell(id_.iCell,id_.iLay,id_.iSubSec,true);
    const HepGeom::Point3D<float> lcoord(xy.first,xy.second,0);
#ifdef DebugLog
    std::cout << "getPosition:: index " << cellIndex << " Local " << xy.first
	      << ":" << xy.second << " ID " << id_.iCell << ":" << id_.iLay 
	      << " Global " << m_cellVec[cellIndex].getPosition(lcoord)
	      << " Cell" << m_cellVec[cellIndex];
#endif
    return m_cellVec[cellIndex].getPosition(lcoord);
  } 
  return GlobalPoint();
}

HGCalGeometry::CornersVec HGCalGeometry::getCorners( const DetId& id ) const {

  HGCalGeometry::CornersVec co (8, GlobalPoint(0,0,0));
  unsigned int cellIndex =  indexFor(id);
  if (cellIndex <  m_cellVec.size()) {
    HGCalTopology::DecodedDetId id_ = topology().decode(id);
    std::pair<float,float> xy = topology().dddConstants().locateCell(id_.iCell,id_.iLay,id_.iSubSec,true);
    float dz = m_cellVec[cellIndex].param()[0];
    float dx = 0.5*m_cellVec[cellIndex].param()[11];
    static const int signx[] = {-1,-1,1,1,-1,-1,1,1};
    static const int signy[] = {-1,1,1,-1,-1,1,1,-1};
    static const int signz[] = {-1,-1,-1,-1,1,1,1,1};
    for (unsigned int i = 0; i != 8; ++i) {
      const HepGeom::Point3D<float> lcoord(xy.first+signx[i]*dx,xy.second+signy[i]*dx,signz[i]*dz);
      co[i] = m_cellVec[cellIndex].getPosition(lcoord);
    }
  }
  return co;
}

DetId HGCalGeometry::getClosestCell( const GlobalPoint& r ) const {
  unsigned int cellIndex = getClosestCellIndex(r);
  if (cellIndex < m_cellVec.size()) {
    const HepGeom::Point3D<float> local = m_cellVec[cellIndex].getLocal(r);
    HGCalTopology::DecodedDetId id_ = topology().decode(m_validGeomIds[cellIndex]);
    std::pair<int,int> kxy = 
      topology().dddConstants().assignCell(local.x(),local.y(),id_.iLay,
					   id_.iSubSec,true);
    id_.iCell   = kxy.second;
    id_.iSubSec = kxy.first;
#ifdef DebugLog
    std::cout << "getClosestCell: local " << local << " Id " << id_.zside 
	      << ":" << id_.iLay << ":" << id_.iSec << ":" << id_.iSubSec
	      << ":" << id_.iCell << " Cell " << m_cellVec[cellIndex];
#endif

    //check if returned cell is valid
    if(id_.iCell>=0) return topology().encode(id_);
  }

  //if not valid or out of bounds return a null DetId
  return DetId();
}

HGCalGeometry::DetIdSet HGCalGeometry::getCells( const GlobalPoint& r, double dR ) const {
   HGCalGeometry::DetIdSet dss;
   return dss;
}
std::string HGCalGeometry::cellElement() const {
  if      (m_subdet == HGCEE)  return "HGCalEE";
  else if (m_subdet == HGCHEF) return "HGCalHEFront";
  else if (m_subdet == HGCHEB) return "HGCalHEBack";
  else                         return "Unknown";
}

unsigned int HGCalGeometry::indexFor(const DetId& id) const {
  unsigned int cellIndex =  m_cellVec.size();
  if (id != DetId()) {
    DetId geoId = (id.subdetId() == HGCEE ? 
		   (DetId)(HGCEEDetId(id).geometryCell()) : 
		   (DetId)(HGCHEDetId(id).geometryCell()));
    cellIndex = topology().detId2denseGeomId(geoId);
#ifdef DebugLog
    std::cout << "indexFor " << std::hex << id.rawId() << ":" << geoId.rawId()
	      << std::dec << " index " << cellIndex << std::endl;
#endif
  }
  return cellIndex;
}

unsigned int HGCalGeometry::sizeForDenseIndex() const {
  return topology().totalGeomModules();
}

const CaloCellGeometry* HGCalGeometry::cellGeomPtr(uint32_t index) const {
  if ((index >= m_cellVec.size()) || (m_validGeomIds[index].rawId() == 0)) 
    return 0;
  const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
#ifdef DebugLog
  std::cout << "cellGeomPtr " << m_cellVec[index];
#endif
  if (0 == cell->param()) return 0;
  return cell;
}

void HGCalGeometry::addValidID(const DetId& id) {
  edm::LogError("HGCalGeom") << "HGCalGeometry::addValidID is not implemented";
}


unsigned int HGCalGeometry::getClosestCellIndex (const GlobalPoint& r) const {

  float phip = r.phi();
  float zp   = r.z();
  unsigned int cellIndex =  m_cellVec.size();
  float dzmin(9999), dphimin(9999), dphi10(0.175);
  for (unsigned int k=0; k<m_cellVec.size(); ++k) {
    float dphi = phip-m_cellVec[k].phiPos();
    while (dphi >   M_PI) dphi -= 2*M_PI;
    while (dphi <= -M_PI) dphi += 2*M_PI;
    if (fabs(dphi) < dphi10) {
      float dz = fabs(zp - m_cellVec[k].getPosition().z());
      if (dz < (dzmin+0.001)) {
	dzmin     = dz;
	if (fabs(dphi) < (dphimin+0.01)) {
	  cellIndex = k;
	  dphimin   = fabs(dphi);
	} else {
	  if (cellIndex >= m_cellVec.size()) cellIndex = k;
	}
      }
    }
  }
#ifdef DebugLog
  std::cout << "getClosestCellIndex::Input " << zp << ":" << phip << " Index "
	    << cellIndex;
  if (cellIndex < m_cellVec.size()) 
    std::cout << " Cell z " << m_cellVec[cellIndex].getPosition().z() << ":"
	      << dzmin << " phi " << m_cellVec[cellIndex].phiPos() << ":"
	      << dphimin;
  std::cout << std::endl;

#endif
  return cellIndex;
}

// FIXME: Change sorting algorithm if needed
namespace
{
  struct rawIdSort
  {
    bool operator()( const DetId& a, const DetId& b )
    {
      return( a.rawId() < b.rawId());
    }
  };
}

void
HGCalGeometry::sortDetIds( void )
{
  m_validIds.shrink_to_fit();
  std::sort( m_validIds.begin(), m_validIds.end(), rawIdSort());
}

void
HGCalGeometry::getSummary( CaloSubdetectorGeometry::TrVec&  trVector,
			   CaloSubdetectorGeometry::IVec&   iVector,
			   CaloSubdetectorGeometry::DimVec& dimVector,
			   CaloSubdetectorGeometry::IVec& dinsVector ) const 
{
  const std::vector<DetId>& ids = getValidDetIds();
  std::cout << ids.size() << " valid ids for " << cellElement() 
	    << std::endl;

  unsigned int numberOfCells = m_topology.totalGeomModules();
  unsigned int numberOfShapes = HGCalGeometry::k_NumberOfShapes;
  unsigned int numberOfParametersPerShape = HGCalGeometry::k_NumberOfParametersPerShape;
  //unsigned int numberOfTransformParms = 0;
  for( auto trItr = m_topology.dddConstants().getFirstTrForm(); 
       trItr != m_topology.dddConstants().getLastTrForm(); ++trItr)
  {
    //++numberOfTransformParms;
  }
  std::cout << "numberOfCells " << numberOfCells << "\nnumberOfTransformParms() " << numberOfTransformParms()
	    << "\nnumberOfShapes " << numberOfShapes << "\nnumberOfParametersPerShape " << HGCalGeometry::k_NumberOfParametersPerShape
	    << "\nparVecVec().size() " << parVecVec().size()
	    << "\nm_cellVec.size() " << m_cellVec.size()
	    << "\nsizeForDenseIndex() " << sizeForDenseIndex() << "\n";

  for( auto it : m_cellVec )
  {
    std::cout << "Center: " <<  it.getPosition()
	      << ", eta " << it.etaPos()
	      << ", phi " << it.phiPos()
	      << std::endl;
  }

  trVector.reserve( numberOfCells*numberOfTransformParms() ) ;
  iVector.reserve( numberOfShapes ==1 ? 1 : numberOfCells ) ;
  dimVector.reserve( numberOfShapes*numberOfParametersPerShape ) ;

  for (ParVecVec::const_iterator ivv (parVecVec().begin()) ; 
       ivv != parVecVec().end() ; ++ivv) {
    const ParVec& pv ( *ivv ) ;
    for (ParVec::const_iterator iv ( pv.begin() ) ; iv != pv.end() ; ++iv) {
      dimVector.push_back( *iv ) ;
    }
  }
  
  for (unsigned int i ( 0 ) ; i < numberOfCells; ++i) {
    Tr3D tr ;
    const CaloCellGeometry* ptr ( cellGeomPtr( i ) ) ;
    
    if (0 != ptr) {

      ptr->getTransform( tr, ( Pt3DVec* ) 0 ) ;

      if( Tr3D() == tr ) { // for preshower there is no rotation
	const GlobalPoint& gp ( ptr->getPosition() ) ; 
	tr = HepGeom::Translate3D( gp.x(), gp.y(), gp.z() ) ;
      }

      const CLHEP::Hep3Vector  tt ( tr.getTranslation() ) ;
      trVector.push_back( tt.x() ) ;
      trVector.push_back( tt.y() ) ;
      trVector.push_back( tt.z() ) ;
      if (6 == numberOfTransformParms()) {
	const CLHEP::HepRotation rr ( tr.getRotation() ) ;
	const ROOT::Math::Transform3D rtr (rr.xx(), rr.xy(), rr.xz(), tt.x(),
					   rr.yx(), rr.yy(), rr.yz(), tt.y(),
					   rr.zx(), rr.zy(), rr.zz(), tt.z());
	ROOT::Math::EulerAngles ea ;
	rtr.GetRotation( ea ) ;
	trVector.push_back( ea.Phi() ) ;
	trVector.push_back( ea.Theta() ) ;
	trVector.push_back( ea.Psi() ) ;
      }

      const CCGFloat* par ( ptr->param() ) ;

      unsigned int ishape ( 9999 ) ;
      for( unsigned int ivv ( 0 ) ; ivv != parVecVec().size() ; ++ivv ) {
	bool ok ( true ) ;
	const CCGFloat* pv ( &(*parVecVec()[ivv].begin() ) ) ;
	for( unsigned int k ( 0 ) ; k != numberOfParametersPerShape ; ++k ) {
	  ok = ok && ( fabs( par[k] - pv[k] ) < 1.e-6 ) ;
	}
	if( ok ) {
	  ishape = ivv ;
	  break ;
	}
      }
      assert( 9999 != ishape ) ;
      
      const unsigned int nn (( numberOfShapes==1) ? (unsigned int)1 : 0 ) ; 
      if( iVector.size() < nn ) iVector.push_back( ishape ) ;
    }
  }
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalGeometry);
