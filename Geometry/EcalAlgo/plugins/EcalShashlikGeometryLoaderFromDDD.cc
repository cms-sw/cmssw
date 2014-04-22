#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalShashlikGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

typedef CaloGeometryLoader< EcalShashlikGeometry > EcalKGL ;
typedef boost::shared_ptr< CaloSubdetectorGeometry > PtrType ;


template <> EcalKGL::CaloGeometryLoader(); 
template <> typename EcalKGL::PtrType EcalKGL::load( const DDCompactView*  cpv        ,
						     const Alignments*     alignments ,
						     const Alignments*     globals       );

template <> void EcalKGL::makeGeometry( const DDCompactView*  cpv        ,
					EcalShashlikGeometry*                    geom       ,
					const Alignments*     alignments ,
					const Alignments*     globals      );

template <> unsigned int EcalKGL::getDetIdForDDDNode( const DDFilteredView& fv );

template <>
void 
EcalKGL::fillGeom( EcalShashlikGeometry*         geom,
		   const EcalKGL::ParmVec&     vv,
		   const HepGeom::Transform3D& tr,
		   const DetId&                id );
template <>
void 
EcalKGL::fillNamedParams( DDFilteredView      fv,
			  EcalShashlikGeometry* geom );

#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.icc"

template class CaloGeometryLoader< EcalShashlikGeometry > ;
typedef CaloCellGeometry::CCGFloat CCGFloat ;
 

template <>
EcalKGL::CaloGeometryLoader() 
{}

template <>
typename EcalKGL::PtrType
EcalKGL::load( const DDCompactView*  cpv        ,
	       const Alignments*     alignments ,
	       const Alignments*     globals       ) 
{
  ShashlikTopology topology (*cpv);
  // allocate geometry
  EcalShashlikGeometry* geom = new EcalShashlikGeometry (topology);
  unsigned int numberOfCells = (unsigned) topology.totalModules () * 2; // both sides
  geom->allocateCorners( numberOfCells ) ;
  geom->allocatePar( EcalShashlikGeometry::k_NumberOfParametersPerShape*EcalShashlikGeometry::k_NumberOfShapes,
		     EcalShashlikGeometry::k_NumberOfParametersPerShape ) ;

  // loop over modules
  DDFilteredView fv ( *cpv ) ;
  DDSpecificsFilter filter; 
  fv.addFilter(filter); // filter nothing. F.R.: Can we filter by LogicalPart?
  fv.reset (); // wolk the entire tree. F.R.: Can optimize here?
  unsigned int counter = 0;
  while (fv.next()) {
    if (fv.logicalPart().name().name() == EcalShashlikGeometry::cellElement()) {
      if (fv.copyNumbers().size() > 2) { // deep enough
	int moduleNr = *(fv.copyNumbers().end()-1);
	int supermoduleNr = *(fv.copyNumbers().end()-2);
	int zSide = *(fv.copyNumbers().end()-3) == 2 ? 1 : -1;
	std::pair<int,int> xy = topology.getXY (supermoduleNr, moduleNr); 
	EKDetId detId (xy.first, xy.second, zSide);
	if (!topology.validXY (xy.first, xy.second)) {
	  edm::LogError("HGCalGeom") << "EcalShashlikGeometryLoaderFromDDD: ignore invalid DDD copy "
				     << fv.logicalPart().name().name() 
				     << " with indexes " 
	    //<< fv.copyNumbers();
				     << "( ..., " << *(fv.copyNumbers().end()-3)
				     << ", " << *(fv.copyNumbers().end()-2)
				     << ", " << *(fv.copyNumbers().end()-1)
				     << ")"
				     << " ix:iy " << xy.first << '/' << xy.second;
	  continue;
	}
	++counter ;
	
	
	const DDSolid& solid ( fv.logicalPart().solid() ) ;
	const ParmVec& parameters ( solid.parameters() ) ;
	
	DD3Vector x, y, z;
	fv.rotation().GetComponents( x, y, z ) ;
	const CLHEP::HepRep3x3 temp( x.X(), y.X(), z.X(),
				     x.Y(), y.Y(), z.Y(),
				     x.Z(), y.Z(), z.Z() );
	const CLHEP::HepRotation hr ( temp );
	const CLHEP::Hep3Vector h3v ( fv.translation().X(),
				      fv.translation().Y(),
				      fv.translation().Z()  ) ;
	const HepGeom::Transform3D ht3d ( hr,          // only scale translation
					  k_ScaleFromDDDtoGeant*h3v ) ;    
	
	const unsigned int which ( geom->alignmentTransformIndexLocal( detId ) ) ;
	
	assert( 0 == alignments ||
		which < alignments->m_align.size() ) ;
	
	const AlignTransform* at ( 0 == alignments ? 0 :
				   &alignments->m_align[ which ] ) ;
	
	assert( 0 == at || ( geom->alignmentTransformIndexLocal( DetId( at->rawId() ) ) == which ) ) ;
	
	const unsigned int gIndex ( geom->alignmentTransformIndexGlobal( detId ) ) ;
	
	const AlignTransform* globalT ( 0 == globals ? 0 :
					( globals->m_align.size() > gIndex ? 
					  &globals->m_align[ gIndex ] : 0 ) ) ;
	
	const HepGeom::Transform3D atr ( 0 == at ? ht3d :
					 ( 0 == globalT ? at->transform()*ht3d :
					   at->transform()*globalT->transform()*ht3d ) ) ;
	
	fillGeom( geom, parameters, atr, detId ) ;
      }
    }
  }
  
  
  assert( counter == numberOfCells ) ;
  
  geom->initializeParms() ;
  return EcalKGL::PtrType (geom);
}

template <>
unsigned int 
EcalKGL::getDetIdForDDDNode( const DDFilteredView& fv )
{return 0;}


template <>
void 
EcalKGL::fillGeom( EcalShashlikGeometry*       geom,
		   const EcalKGL::ParmVec&     vv,
		   const HepGeom::Transform3D& tr,
		   const DetId&                id )
{
   std::vector<CCGFloat> pv ;
   pv.reserve( vv.size() ) ;
   for( unsigned int i ( 0 ) ; i != vv.size() ; ++i )
   {
      const CCGFloat factor ( 1==i || 2==i || 6==i || 10==i ? 1 : k_ScaleFromDDDtoGeant ) ;
      pv.push_back( factor*vv[i] ) ;
   }

   std::vector<GlobalPoint> corners (8);

   TruncatedPyramid::createCorners( pv, tr, corners ) ;

   const CCGFloat* parmPtr ( CaloCellGeometry::getParmPtr( pv, 
							   geom->parMgr(), 
							   geom->parVecVec() ) ) ;

   const GlobalPoint front ( 0.25*( corners[0].x() + 
				    corners[1].x() + 
				    corners[2].x() + 
				    corners[3].x()   ),
			     0.25*( corners[0].y() + 
				    corners[1].y() + 
				    corners[2].y() + 
				    corners[3].y()   ),
			     0.25*( corners[0].z() + 
				    corners[1].z() + 
				    corners[2].z() + 
				    corners[3].z()   ) ) ;
   
   const GlobalPoint back  ( 0.25*( corners[4].x() + 
				    corners[5].x() + 
				    corners[6].x() + 
				    corners[7].x()   ),
			     0.25*( corners[4].y() + 
				    corners[5].y() + 
				    corners[6].y() + 
				    corners[7].y()   ),
			     0.25*( corners[4].z() + 
				    corners[5].z() + 
				    corners[6].z() + 
				    corners[7].z()   ) ) ;

   geom->newCell( front, back, corners[0],
		  parmPtr, 
		  id ) ;
}

template <>
void 
EcalKGL::fillNamedParams ( DDFilteredView      fv,
			  EcalShashlikGeometry* geom )
{}
