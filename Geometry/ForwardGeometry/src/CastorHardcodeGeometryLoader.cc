#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorHardcodeGeometryLoader.h"
#include "Geometry/ForwardGeometry/interface/IdealCastorTrapezoid.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "Geometry/ForwardGeometry/src/CastorGeometryData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <utility>

typedef CaloCellGeometry::CCGFloat CCGFloat ;

CastorHardcodeGeometryLoader::CastorHardcodeGeometryLoader() :
   theTopology ( new CastorTopology ) ,
   extTopology ( theTopology )
{
   init();
}

CastorHardcodeGeometryLoader::CastorHardcodeGeometryLoader( const CastorTopology& ht ) : 
   theTopology( nullptr ) , 
   extTopology ( &ht )
{
   init();
}

void CastorHardcodeGeometryLoader::init() 
{
   theEMSectiondX = 2.*dXEMPlate;
   theEMSectiondY = 2.*dYEMPlate;
   theEMSectiondZ = 101.0;
   theHADSectiondX = 2.*dXHADPlate; 
   theHADSectiondY = 2.*dYHADPlate;
   theHADSectiondZ = 1212.;
}

std::unique_ptr<CaloSubdetectorGeometry> 
CastorHardcodeGeometryLoader::load( DetId::Detector /*det*/, 
				    int             subdet)
{
   std::unique_ptr<CaloSubdetectorGeometry> hg(new CastorGeometry( extTopology ));
   if( subdet == HcalCastorDetId::SubdetectorId )
   {
      fill( HcalCastorDetId::EM,  hg.get() ) ;
      fill( HcalCastorDetId::HAD, hg.get() ) ;
   }
   return hg;
}

std::unique_ptr<CaloSubdetectorGeometry> 
CastorHardcodeGeometryLoader::load() 
{
   std::unique_ptr<CaloSubdetectorGeometry> hg
      ( new CastorGeometry( extTopology ) ) ;
   fill( HcalCastorDetId::EM,  hg.get() ) ;
   fill( HcalCastorDetId::HAD, hg.get() ) ;
   return hg;
}

void 
CastorHardcodeGeometryLoader::fill( HcalCastorDetId::Section section , 
				    CaloSubdetectorGeometry* geom      ) 
{
   if( geom->cornersMgr() == nullptr ) geom->allocateCorners(
      HcalCastorDetId::kSizeForDenseIndexing ) ;
   if( geom->parMgr()     == nullptr ) geom->allocatePar( 
      CastorGeometry::k_NumberOfShapes*
      CastorGeometry::k_NumberOfParametersPerShape,
      CastorGeometry::k_NumberOfParametersPerShape ) ;

   // start by making the new HcalDetIds
   std::vector<HcalCastorDetId> castorIds ;

   const int firstCell ( extTopology->firstCell( section ) ) ;
   const int lastCell  ( extTopology->lastCell(  section ) );

   for( int imodule ( firstCell ) ; imodule <= lastCell ; ++imodule ) 
   {
      for( int isector ( 1 ) ;
	   isector <= HcalCastorDetId::kNumberSectorsPerEnd ; ++isector ) 
      {
	 const HcalCastorDetId id ( section, false, isector, imodule ) ;
	 if( extTopology->valid( id ) )  castorIds.emplace_back( id ) ;
      }
   }
//   edm::LogInfo("CastorHardcodeGeometry") 
//      << "Number of Castor DetIds made: " << section 
//      << " " << castorIds.size();
 
   // for each new HcalCastorDetId, make a CaloCellGeometry

   for( std::vector<HcalCastorDetId>::const_iterator 
	   castorIdItr ( castorIds.begin() ) ;
	castorIdItr != castorIds.end() ; ++castorIdItr )
   {
      makeCell( *castorIdItr, geom ) ;
   }
}

void
CastorHardcodeGeometryLoader::makeCell( const HcalCastorDetId&   detId , 
					CaloSubdetectorGeometry* geom   ) const 
{
   const double                   zside   ( 1.0*detId.zside()  ) ;
   const HcalCastorDetId::Section section ( detId.section()    ) ;
   const int                      module  ( detId.module()     ) ;
   const int                      isect   ( detId.sector()     ) ;
   const double                   sector  ( 1.0*isect          ) ;

// length units are cm


   const double sign ( 0 == isect%2 ? -1 : 1 ) ;

//********* HERE ARE HARDWIRED GEOMETRY NUMBERS ACTUALLY USED *****

   static const double an     ( atan( 1.)); //angle of cant w.r.t. beam
   static const double can    ( cos( an ));
   static const double san    ( sin( an ));
   static const double dxlEM  ( 1.55/2. ) ; //halflength of side near beam
   static const double dxhEM  ( 5.73/2. ) ; //halflength of side away from beam
   static const double dhEM   ( 14.26/2. ); //halflength of 2nd longest side
   static const double dR     ( 0.1 + 2.*dhEM*san*dxlEM/( dxhEM-dxlEM ) ) ;
   static const double dhHAD  ( 19.88/2. ); //halflength of 2nd longest side 

   static const double dxhHAD ( dxhEM*( 2.*dhHAD*san + dR )/
				( 2.*dhEM*san + dR ) ) ; //halflength of side away from beam
   static const double zm     ( 1439.0  ) ;  //z of start of EM
   static const double dzEM   ( 5.45/2 ) ;   // halflength in z of EM
   static const double dzHAD  ( 10.075/2 ) ; // halflength in z of HAD

//*****************************************************************

   const double dxl ( sign*dxlEM ) ; // same for EM and HAD

   const double dxh ( sign*( section == HcalCastorDetId::EM ?
			     dxhEM : dxhHAD ) ) ;
   const double dh  ( section == HcalCastorDetId::EM ?
		      dhEM : dhHAD ) ;
   const double dz  ( section == HcalCastorDetId::EM ?
		      dzEM : dzHAD ) ;

   const double delz  ( dh*can ) ;
   const double dy  ( dh*san ) ;
   const double dx  ( ( dxl + dxh )/2. ) ;
   const double leg ( dR + dy ) ;
   const double len ( sqrt( leg*leg + dx*dx ) ) ;

   static const double dphi 
      ( 2.*M_PI/(1.0*HcalCastorDetId::kNumberSectorsPerEnd ) ) ;

   const double fphi ( atan( dx/( dR + dy ) ) ) ;

   const double phi  ( 0==isect%2 ? (sector-1.)*dphi - fphi :
		       sector*dphi - fphi ) ;

   const double sphi ( sin( phi ) ) ;
   const double cphi ( cos( phi ) ) ;

   const double xc ( len*cphi ) ;
   const double yc ( len*sphi ) ; 
   const double zc ( zside*( zm + delz + 
			     ( module<3 ? ( 1.*module - 1.0 )*2.*dzEM :
			       4.*dzEM +  ( 1.*(module-2) - 1 )*2.*dzHAD ) ) ) ;

   const GlobalPoint fc ( xc, yc, zc ) ;

   std::vector<CCGFloat> zz ;
   zz.reserve( CastorGeometry::k_NumberOfParametersPerShape ) ;
   zz.emplace_back( dxl ) ;
   zz.emplace_back( dxh ) ;
   zz.emplace_back( dh ) ;
   zz.emplace_back( dz ) ;
   zz.emplace_back( an ) ;
   zz.emplace_back( dR ) ;

   geom->newCell( fc, fc, fc, 
		  CaloCellGeometry::getParmPtr( zz, 
						geom->parMgr(), 
						geom->parVecVec() ),
		  detId ) ;
}


