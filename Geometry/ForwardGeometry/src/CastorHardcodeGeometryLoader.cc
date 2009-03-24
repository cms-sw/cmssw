#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorHardcodeGeometryLoader.h"
#include "Geometry/ForwardGeometry/interface/IdealCastorTrapezoid.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "Geometry/ForwardGeometry/src/CastorGeometryData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

CastorHardcodeGeometryLoader::CastorHardcodeGeometryLoader() 
{
   init();
}

CastorHardcodeGeometryLoader::CastorHardcodeGeometryLoader( 
   const CastorTopology& ht ) : 
   theTopology( ht ) 
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

std::auto_ptr<CaloSubdetectorGeometry> 
CastorHardcodeGeometryLoader::load( DetId::Detector det, 
				    int             subdet)
{
   std::auto_ptr<CaloSubdetectorGeometry> hg(new CastorGeometry(&theTopology));
   if( subdet == HcalCastorDetId::SubdetectorId )
   {
      fill( HcalCastorDetId::EM,  hg.get() ) ;
      fill( HcalCastorDetId::HAD, hg.get() ) ;
   }
   return hg;
}

std::auto_ptr<CaloSubdetectorGeometry> 
CastorHardcodeGeometryLoader::load() 
{
   std::auto_ptr<CaloSubdetectorGeometry> hg
      ( new CastorGeometry( &theTopology ) ) ;
   fill( HcalCastorDetId::EM,  hg.get() ) ;
   fill( HcalCastorDetId::HAD, hg.get() ) ;
   return hg;
}

void 
CastorHardcodeGeometryLoader::fill( HcalCastorDetId::Section section , 
				    CaloSubdetectorGeometry* geom      ) 
{
   if( geom->cornersMgr() == 0 ) geom->allocateCorners(
      HcalCastorDetId::kSizeForDenseIndexing ) ;
   if( geom->parMgr()     == 0 ) geom->allocatePar( 
      CastorGeometry::k_NumberOfShapes*
      CastorGeometry::k_NumberOfParametersPerShape,
      CastorGeometry::k_NumberOfParametersPerShape ) ;

   // start by making the new HcalDetIds
   std::vector<HcalCastorDetId> castorIds ;

   const int firstCell ( theTopology.firstCell( section ) ) ;
   const int lastCell  ( theTopology.lastCell(  section ) );

   for( int imodule ( firstCell ) ; imodule <= lastCell ; ++imodule ) 
   {
      for( int isector ( 1 ) ;
	   isector <= HcalCastorDetId::kNumberSectorsPerEnd ; ++isector ) 
      {
	 const HcalCastorDetId id ( section, false, isector, imodule ) ;
	 if( theTopology.valid( id ) )  castorIds.push_back( id ) ;
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
      geom->addCell( *castorIdItr, makeCell( *castorIdItr, geom ) ) ;
   }
}

CaloCellGeometry*
CastorHardcodeGeometryLoader::makeCell( const HcalCastorDetId&   detId , 
					CaloSubdetectorGeometry* geom   ) const 
{
   const double                   zside   ( 1.0*detId.zside()  ) ;
   const HcalCastorDetId::Section section ( detId.section()    ) ;
   const int                      module  ( detId.module()     ) ;
   const int                      isect   ( detId.sector()     ) ;
   const double                   sector  ( 1.0*isect          ) ;

// length units are cm

   const unsigned int iz ( section == HcalCastorDetId::EM ? module : 2 + module ) ;

   const double sign ( 0 == isect%2 ? -1 : 1 ) ;
   const double dxl ( sign*1.55/2. ) ; 
   double dxh ( 0 ) ; 
   double dh ( 0 ) ;
   
   static const double zm    ( 1439.0 ) ;
   static const double an    ( atan( 1.0 ) ) ;
   static const double can   ( cos( an ) ) ;
   static const double san   ( sin( an ) ) ;
   static const double dR    ( 7.48 ) ;
   static const double dzEM  ( 5.45 ) ;
   static const double dzHAD ( 10.075 ) ;

   static const double dphi 
      ( 2.*M_PI/(1.0*HcalCastorDetId::kNumberSectorsPerEnd ) ) ;

   const double phi  ( ( sector - 0.5 )*dphi ) ;
   const double sphi ( sin( phi ) ) ;
   const double cphi ( cos( phi ) ) ;

   if( section == HcalCastorDetId::EM )
   {
      dxh = sign*5.73 ;
      dh  = 14.26 ;
   }
   else
   {
      dxh = sign*7.37;
      dh  = 19.88 ;
   }

   const double dz  ( dh*can ) ;
   const double dy  ( dh*san ) ;
   const double dx  ( ( dxl + dxh )/2. ) ;
   const double leg ( dR + dy ) ;
   const double len ( sqrt( leg*leg + dx*dx ) ) ;

   const double xc ( len*cphi ) ;
   const double yc ( len*sphi ) ; 
   const double zc ( zside*( zm + dz + 
			     ( iz<3 ? ( 1.*module - 1.0 )*dzEM :
			       2.*dzEM +  ( 1.*module - 1 )*dzHAD ) ) ) ;

   const GlobalPoint fc ( xc, yc, zc ) ;

   std::vector<double> zz ;
   zz.reserve( CastorGeometry::k_NumberOfParametersPerShape ) ;
   zz.push_back( dxl ) ;
   zz.push_back( dxh ) ;
   zz.push_back( dh ) ;
   zz.push_back( dz ) ;
   zz.push_back( an ) ;
   zz.push_back( dR ) ;

   return new calogeom::IdealCastorTrapezoid( 
      fc, 
      geom->cornersMgr(),
      CaloCellGeometry::getParmPtr( zz, 
				    geom->parMgr(), 
				    geom->parVecVec() ) );
}


