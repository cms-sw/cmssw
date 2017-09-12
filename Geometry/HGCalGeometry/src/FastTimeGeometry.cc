#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cmath>

#include <Math/Transform3D.h>
#include <Math/EulerAngles.h>

typedef CaloCellGeometry::Tr3D Tr3D;
typedef std::vector<float> ParmVec;

FastTimeGeometry::FastTimeGeometry( const FastTimeTopology& topology_ )
  : m_topology(topology_),
    m_cellVec(topology_.totalGeomModules()),
    m_validGeomIds(topology_.totalGeomModules()),
    m_Type(topology_.detectorType()),
    m_subdet(topology_.subDetector()) {
  
  m_validIds.reserve(topology().totalModules());
#ifdef EDM_ML_DEBUG
  std::cout << "Expected total # of Geometry Modules " 
	    << topology().totalGeomModules() << std::endl;
#endif
}

FastTimeGeometry::~FastTimeGeometry() { }

void FastTimeGeometry::fillNamedParams (DDFilteredView fv) {}

void FastTimeGeometry::initializeParms() {
}

void FastTimeGeometry::localCorners(Pt3DVec&        lc,
				    const CCGFloat* pv,
				    unsigned int    i,
				    Pt3D&           ref) {
  FlatTrd::localCorners( lc, pv, ref ) ;
}

void FastTimeGeometry::newCell( const GlobalPoint& f1 ,
				const GlobalPoint& f2 ,
				const GlobalPoint& f3 ,
				const CCGFloat*    parm ,
				const DetId&       detId ) {

  FastTimeTopology::DecodedDetId id = topology().decode(detId);
  DetId geomId = (DetId)(FastTimeDetId(detId).geometryCell());
  int nEtaZ = topology().dddConstants().numberEtaZ(m_Type);
  int nPhi  = topology().dddConstants().numberPhi(m_Type);

  const uint32_t cellIndex (topology().detId2denseGeomId(detId));

  m_cellVec.at( cellIndex ) = FlatTrd( cornersMgr(), f1, f2, f3, parm ) ;
  m_validGeomIds.at( cellIndex ) = geomId ;

#ifdef EDM_ML_DEBUG
  unsigned int nOld = m_validIds.size();
#endif
  for (int etaZ = 1; etaZ <= nEtaZ; ++etaZ) {
    id.iEtaZ = etaZ;
    for (int phi = 1; phi <= nPhi; ++phi) {
      id.iPhi  = phi;
      DetId idc = topology().encode(id);
      if (topology().valid(idc)) {
	m_validIds.emplace_back(idc);
      }
    }
  }

#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeGeometry::newCell-> [" << cellIndex << "]"
	    << " front:" << f1.x() << '/' << f1.y() << '/' << f1.z() 
     	    << " back:" <<  f2.x() << '/' << f2.y() << '/' << f2.z()
	    << " eta|phi " << m_cellVec[cellIndex].etaPos() << ":"
	    << m_cellVec[cellIndex].phiPos() << " id:" << FastTimeDetId(detId)
	    << " with valid DetId from " << nOld << " to " << m_validIds.size()
 	    << std::endl; 
  std::cout << "Cell[" << cellIndex << "] " << std::hex << geomId.rawId() 
	    << ":"  << m_validGeomIds[cellIndex].rawId() << std::dec
	    << std::endl;
#endif
}

const CaloCellGeometry* FastTimeGeometry::getGeometry(const DetId& id) const {

  if (id == DetId()) return nullptr; // nothing to get
  DetId geoId = (DetId)(FastTimeDetId(id).geometryCell());
  const uint32_t cellIndex (topology().detId2denseGeomId(geoId));
  return cellGeomPtr (cellIndex);
}

GlobalPoint FastTimeGeometry::getPosition(const DetId& id) const {

  FastTimeDetId id_ = FastTimeDetId(id);
  auto pos = topology().dddConstants().getPosition(m_Type,id_.ieta(),id_.iphi(),id_.zside());
  return GlobalPoint(0.1*pos.x(),0.1*pos.y(),0.1*pos.z());
}

FastTimeGeometry::CornersVec FastTimeGeometry::getCorners(const DetId& id) const {

  FastTimeDetId id_ = FastTimeDetId(id);
  auto corners = topology().dddConstants().getCorners(m_Type,id_.ieta(),id_.iphi(),id_.zside());
  FastTimeGeometry::CornersVec out;
  for( const auto& corner : corners ) {
    out.emplace_back(0.1*corner.x(),0.1*corner.y(),0.1*corner.z());
  }
  return out;
}

DetId FastTimeGeometry::getClosestCell(const GlobalPoint& r) const {
  int zside = (r.z() > 0) ? 1 : -1;
  std::pair<int,int> etaZPhi;
  if (m_Type == 1) {
    double zz = (zside > 0) ? r.z() : -r.z();
    etaZPhi = topology().dddConstants().getZPhi(zz,r.phi());
  } else {
    double phi = (zside > 0) ? r.phi() : atan2(r.y(),-r.x());
    etaZPhi = topology().dddConstants().getEtaPhi(r.perp(),phi);
  }
  FastTimeDetId id = FastTimeDetId(m_Type,etaZPhi.first,etaZPhi.second,zside);
#ifdef EDM_ML_DEBUG
  std::cout << "getClosestCell: for (" << r.x() << ", " << r.y() << ", "
	    << r.z() << ")  Id " << id.type() << ":" << id.zside() << ":" 
	    << id.ieta() << ":" << id.iphi() << std::endl;
#endif

  return (topology().valid(id) ? DetId(id) : DetId());
}

FastTimeGeometry::DetIdSet FastTimeGeometry::getCells(const GlobalPoint& r, double dR ) const {
   FastTimeGeometry::DetIdSet dss;
   return dss;
}

std::string FastTimeGeometry::cellElement() const {
  if      (m_Type == 1) return "FastTimeBarrel";
  else if (m_Type == 2) return "FastTimeEndcap";
  else                  return "Unknown";
}

unsigned int FastTimeGeometry::indexFor(const DetId& id) const {
  unsigned int cellIndex =  m_cellVec.size();
  if (id != DetId()) {
    DetId geoId = (DetId)(FastTimeDetId(id).geometryCell());
    cellIndex = topology().detId2denseGeomId(geoId);
#ifdef EDM_ML_DEBUG
    std::cout << "indexFor " << std::hex << id.rawId() << ":" << geoId.rawId()
	      << std::dec << " index " << cellIndex << std::endl;
#endif
  }
  return cellIndex;
}

unsigned int FastTimeGeometry::sizeForDenseIndex() const {
  return topology().totalGeomModules();
}

const CaloCellGeometry* FastTimeGeometry::cellGeomPtr(uint32_t index) const {
  if ((index >= m_cellVec.size()) || (m_validGeomIds[index].rawId() == 0)) 
    return nullptr;
  const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
#ifdef EDM_ML_DEBUG
  //  std::cout << "cellGeomPtr " << m_cellVec[index];
#endif
  if (nullptr == cell->param()) return nullptr;
  return cell;
}

void FastTimeGeometry::addValidID(const DetId& id) {
  edm::LogError("FastTimeGeom") << "FastTimeGeometry::addValidID is not implemented";
}


// FIXME: Change sorting algorithm if needed
namespace {
  struct rawIdSort {
    bool operator()( const DetId& a, const DetId& b ) {
      return( a.rawId() < b.rawId());
    }
  };
}

void FastTimeGeometry::sortDetIds( void ) {
  m_validIds.shrink_to_fit();
  std::sort( m_validIds.begin(), m_validIds.end(), rawIdSort());
}

void FastTimeGeometry::getSummary(CaloSubdetectorGeometry::TrVec&  trVector,
				  CaloSubdetectorGeometry::IVec&   iVector,
				  CaloSubdetectorGeometry::DimVec& dimVector,
				  CaloSubdetectorGeometry::IVec& dinsVector ) const {

  unsigned int numberOfCells = topology().totalGeomModules(); // total Geom Modules both sides
  unsigned int numberOfShapes = FastTimeGeometry::k_NumberOfShapes;
  unsigned int numberOfParametersPerShape = FastTimeGeometry::k_NumberOfParametersPerShape;

  trVector.reserve( numberOfCells * numberOfTransformParms());
  iVector.reserve( numberOfCells );
  dimVector.reserve( numberOfShapes * numberOfParametersPerShape );
  dinsVector.reserve( numberOfCells );
  
  for (unsigned int k=0; k <topology().totalGeomModules(); ++k) {
    ParmVec params( FastTimeGeometry::k_NumberOfParametersPerShape, 0 );
    params[0] = topology().dddConstants().getZHalf(m_Type);
    params[1] = params[2] = 0;
    params[3] = params[7] = topology().dddConstants().getRin(m_Type);
    params[4] = params[8] = topology().dddConstants().getRout(m_Type);
    params[5] = params[9] = topology().dddConstants().getRout(m_Type);
    params[6] = params[10]= 0;
    params[11]= (k == 0) ? 1.0 : -1.0;
    dimVector.insert( dimVector.end(), params.begin(), params.end());
  }
  
  for (unsigned int i( 0 ); i < numberOfCells; ++i) {
    DetId detId = m_validGeomIds[i];
    dinsVector.emplace_back( topology().detId2denseGeomId( detId ));
    iVector.emplace_back(1);
    
    Tr3D tr;
    const CaloCellGeometry* ptr( cellGeomPtr( i ));
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

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(FastTimeGeometry);
