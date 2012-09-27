#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

typedef CaloGeometryLoader< EcalPreshowerGeometry > EcalPGL ;

template <>
void 
EcalPGL::fillGeom( EcalPreshowerGeometry*      geom ,
		   const EcalPGL::ParmVec&     pv ,
		   const HepGeom::Transform3D& tr ,
		   const DetId&                id     );
template <>
void 
EcalPGL::fillNamedParams( DDFilteredView         /*fv*/   ,
			  EcalPreshowerGeometry* /*geom*/  );

#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.icc"

template class CaloGeometryLoader< EcalPreshowerGeometry > ;
typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;

template <>
void 
EcalPGL::fillGeom( EcalPreshowerGeometry*      geom ,
		   const EcalPGL::ParmVec&     pv ,
		   const HepGeom::Transform3D& tr ,
		   const DetId&                id     )
{
   std::vector<CCGFloat> vv ;
   vv.reserve( pv.size() + 1 ) ;
   for( unsigned int i ( 0 ) ; i != pv.size() ; ++i )
   {
      vv.push_back( k_ScaleFromDDDtoGeant*pv[i] ) ;
   }

   const Pt3D cor1 (  vv[0],  vv[1], vv[2] ) ;
   const Pt3D cor2 ( -vv[0],  vv[1], vv[2] ) ;
   const Pt3D cor3 (  vv[0], -vv[1], vv[2] ) ;

   const CCGFloat z1 ( Pt3D( tr*cor1 ).z() ) ;
   const CCGFloat z2 ( Pt3D( tr*cor2 ).z() ) ;
   const CCGFloat z3 ( Pt3D( tr*cor3 ).z() ) ;

   const CCGFloat y1 ( Pt3D( tr*cor3 ).y() ) ;
//   const CCGFloat y3 ( Pt3D( tr*cor3 ).y() ) ;

   const CCGFloat x1 ( Pt3D( tr*cor3 ).x() ) ;
//   const CCGFloat x2 ( Pt3D( tr*cor3 ).x() ) ;

   const CCGFloat zdif ( 0.00001 > fabs( z1 - z2 ) ? 
		       ( y1>0 ? +1.0 : -1.0 )*( z1 - z3 ) : 
		       ( x1>0 ? +1.0 : -1.0 )*( z1 - z2 ) ) ;

   const CCGFloat tilt ( asin( 0.5*zdif/( vv[1]>vv[0] ? vv[1] : vv[0] ) ) ) ;

   vv.push_back( tilt ) ;

   const CCGFloat* pP ( CaloCellGeometry::getParmPtr( vv, 
						      geom->parMgr(), 
						      geom->parVecVec() ) ) ;
   
   const Pt3D ctr ( tr*Pt3D(0,0,0) ) ;

   const GlobalPoint refPoint ( ctr.x(), ctr.y(), ctr.z() ) ;

   geom->newCell( refPoint, refPoint, refPoint,
		  pP,
		  id );
}

template <>
void 
EcalPGL::fillNamedParams( DDFilteredView         /*fv*/   ,
			  EcalPreshowerGeometry* /*geom*/  )
{
   // nothing yet for preshower
}

