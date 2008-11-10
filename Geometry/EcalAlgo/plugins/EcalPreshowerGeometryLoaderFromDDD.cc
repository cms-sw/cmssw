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

using namespace std;

typedef CaloGeometryLoader< EcalPreshowerGeometry > EcalPGL ;


template <>
void 
EcalPGL::fillGeom( EcalPreshowerGeometry*  geom ,
		   const EcalPGL::ParmVec& pv ,
		   const HepTransform3D&   tr ,
		   const DetId&            id     )
{
   std::vector<double> vv ;
   vv.reserve( pv.size() ) ;
   for( unsigned int i ( 0 ) ; i != pv.size() ; ++i )
   {
      vv.push_back( k_ScaleFromDDDtoGeant*pv[i] ) ;
   }
   const double* pP ( CaloCellGeometry::getParmPtr( vv, 
						    geom->parMgr(), 
						    geom->parVecVec() ) ) ;
   
   const HepPoint3D ctr ( tr*HepPoint3D(0,0,0) ) ;

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

