#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.icc"

template class CaloGeometryLoader< EcalPreshowerGeometry > ;

#include "DetectorDescription/Core/interface/DDFilteredView.h"
//#include "DetectorDescription/Core/interface/DDInit.h"


#include <iostream>
#include <vector>

typedef CaloGeometryLoader< EcalPreshowerGeometry > EcalPGL ;


template <>
void 
EcalPGL::fillGeom( EcalPreshowerGeometry*  geom ,
		   const EcalPGL::ParmVec& pv ,
		   const HepGeom::Transform3D&   tr ,
		   const DetId&            id     )
{
   std::vector<double> vv ;
   vv.reserve( pv.size() + 1 ) ;
   for( unsigned int i ( 0 ) ; i != pv.size() ; ++i )
   {
      vv.push_back( k_ScaleFromDDDtoGeant*pv[i] ) ;
   }

   const HepGeom::Point3D<double> cor1 (  vv[0],  vv[1], vv[2] ) ;
   const HepGeom::Point3D<double> cor2 ( -vv[0],  vv[1], vv[2] ) ;
   const HepGeom::Point3D<double> cor3 (  vv[0], -vv[1], vv[2] ) ;

   const double z1 ( HepGeom::Point3D<double>( tr*cor1 ).z() ) ;
   const double z2 ( HepGeom::Point3D<double>( tr*cor2 ).z() ) ;
   const double z3 ( HepGeom::Point3D<double>( tr*cor3 ).z() ) ;

   const double y1 ( HepGeom::Point3D<double>( tr*cor3 ).y() ) ;
//   const double y3 ( HepGeom::Point3D<double>( tr*cor3 ).y() ) ;

   const double x1 ( HepGeom::Point3D<double>( tr*cor3 ).x() ) ;
//   const double x2 ( HepGeom::Point3D<double>( tr*cor3 ).x() ) ;

   const double zdif ( 0.00001 > fabs( z1 - z2 ) ? 
		       ( y1>0 ? +1.0 : -1.0 )*( z1 - z3 ) : 
		       ( x1>0 ? +1.0 : -1.0 )*( z1 - z2 ) ) ;

   const double tilt ( asin( 0.5*zdif/( vv[1]>vv[0] ? vv[1] : vv[0] ) ) ) ;

   vv.push_back( tilt ) ;

//   std::cout<<" For strip at z="<<z1<<", "
//	    << ( vv[1]>vv[0] ? "x" : "y" ) << ", tilt="
//	    << tilt*180./M_PI << std::endl ;

   const double* pP ( CaloCellGeometry::getParmPtr( vv, 
						    geom->parMgr(), 
						    geom->parVecVec() ) ) ;
   
   const HepGeom::Point3D<double>  ctr ( tr*HepGeom::Point3D<double> (0,0,0) ) ;

   const GlobalPoint refPoint ( ctr.x(), ctr.y(), ctr.z() ) ;

   PreshowerStrip* cell ( new PreshowerStrip( refPoint,
					      geom->cornersMgr(),
					      pP ) ) ;

   geom->addCell( id, cell );
}

template <>
void 
EcalPGL::fillNamedParams( DDFilteredView         fv   ,
						     EcalPreshowerGeometry* geom  )
{
   // nothing yet for preshower
}

