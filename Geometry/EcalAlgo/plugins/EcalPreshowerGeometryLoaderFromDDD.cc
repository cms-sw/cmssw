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
unsigned int 
EcalPGL::whichTransform( const DetId& id )  const
{
   return 0 ;
}


template <>
void 
EcalPGL::fillGeom( EcalPreshowerGeometry*  geom ,
		   const EcalPGL::ParmVec& pv ,
		   const HepTransform3D&   tr ,
		   const DetId&            id     )
{
   if( geom->parMgr()     == 0 ) geom->allocatePar( 2, pv.size() ) ;

   std::vector<float> vv ;
   vv.reserve( pv.size() ) ;
   for( unsigned int i ( 0 ) ; i != pv.size() ; ++i )
   {
      vv.push_back( CaloCellGeometry::k_ScaleFromDDDtoGeant*pv[i] ) ;
   }
   const float* pP ( CaloCellGeometry::getParmPtr( vv, 
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
EcalPGL::extraStuff( EcalPreshowerGeometry* geom )
{
   typedef CaloSubdetectorGeometry::CellCont Cont ;
   unsigned int n1 ( 0 ) ;
   unsigned int n2 ( 0 ) ;
   float z1 ( 0 ) ;
   float z2 ( 0 ) ;
   const Cont& con ( geom->cellGeometries() ) ;
   for( Cont::const_iterator i ( con.begin() ) ; i != con.end() ; ++i )
   {
      const ESDetId esid ( i->first ) ;
      if( 1 == esid.plane() )
      {
	 z1 += fabs( i->second->getPosition().z() ) ;
	 ++n1 ;
      }
      if( 2 == esid.plane() )
      {
	 z2 += fabs( i->second->getPosition().z() ) ;
	 ++n2 ;
      }
//      if( 0 == z1 && 1 == esid.plane() ) z1 = fabs( i->second->getPosition().z() ) ;
//      if( 0 == z2 && 2 == esid.plane() ) z2 = fabs( i->second->getPosition().z() ) ;
//      if( 0 != z1 && 0 != z2 ) break ;
   }
   assert( 0 != n1 && 0 != n2 ) ;
   z1 /= (1.*n1) ;
   z2 /= (1.*n2) ;
   assert( 0 != z1 && 0 != z2 ) ;
   geom->setzPlanes( z1, z2 ) ;
}

template <>
void 
EcalPGL::fillNamedParams( DDFilteredView         fv   ,
						     EcalPreshowerGeometry* geom  )
{
   // nothing yet for preshower
}

