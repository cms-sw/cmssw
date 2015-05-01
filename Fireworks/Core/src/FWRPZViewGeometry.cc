// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZViewGeometry
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Thu Mar 25 20:33:06 CET 2010
//

// system include files
#include <iostream>
#include <cassert>

// user include files
#include "TGeoBBox.h"

#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEveScene.h"
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"
#include "TEveManager.h"
#include "TEveProjectionManager.h"

#include "Fireworks/Core/interface/FWRPZViewGeometry.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

//
//
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWRPZViewGeometry::FWRPZViewGeometry(const fireworks::Context& context):
   FWViewGeometryList(context),

   m_rhoPhiGeo(0),
   m_rhoZGeo(0),

   m_pixelBarrelElements(0),
   m_pixelEndcapElements(0),
   m_trackerBarrelElements(0),
   m_trackerEndcapElements(0),
   m_rpcEndcapElements(0),
   m_GEMElements(0),
   m_ME0Elements(0)
{
   SetElementName("RPZGeomShared");
}

// FWRPZViewGeometry::FWRPZViewGeometry(const FWRPZViewGeometry& rhs)
// {
//    // do actual copying here;
// }

FWRPZViewGeometry::~FWRPZViewGeometry()
{
   m_rhoPhiGeo->DecDenyDestroy();
   m_rhoZGeo->DecDenyDestroy();
}

//______________________________________________________________________________

void
FWRPZViewGeometry::initStdGeoElements(const FWViewType::EType type)
{
   assert(m_geom != 0);

   if (type == FWViewType::kRhoZ)
   {
      AddElement(makeMuonGeometryRhoZ());
      AddElement(makeCaloOutlineRhoZ());
   }
   else 
   {
      AddElement(makeMuonGeometryRhoPhi());
      AddElement(makeCaloOutlineRhoPhi());
   }
}

//______________________________________________________________________________


TEveElement*
FWRPZViewGeometry::makeCaloOutlineRhoZ()
{
   using namespace fireworks;

   float ri = m_context.caloZ2()*tan(2*atan(exp(-m_context.caloMaxEta())));

   TEveStraightLineSet* el = new TEveStraightLineSet( "TrackerRhoZoutline" );
   el->SetPickable(kFALSE);
   addToCompound(el, kFWTrackerBarrelColorIndex, false);

   el->AddLine(0,  m_context.caloR1(), -m_context.caloZ1(), 0,  m_context.caloR1(),  m_context.caloZ1());
   el->AddLine(0, -m_context.caloR1(),  m_context.caloZ1(), 0, -m_context.caloR1(), -m_context.caloZ1());

   el->AddLine(0, -m_context.caloR2(),   m_context.caloZ2(), 0,  -ri,   m_context.caloZ2());
   el->AddLine(0, ri,  m_context.caloZ2(), 0,  m_context.caloR2(), m_context.caloZ2());

   el->AddLine(0, -m_context.caloR2(),   -m_context.caloZ2(), 0,  -ri,   -m_context.caloZ2());
   el->AddLine(0, ri,  -m_context.caloZ2(), 0,  m_context.caloR2(), -m_context.caloZ2());
 
   return el;
}

TEveElement*
FWRPZViewGeometry::makeCaloOutlineRhoPhi()
{ 
   TEveStraightLineSet* el = new TEveStraightLineSet( "TrackerRhoPhi" );
   addToCompound(el, kFWTrackerBarrelColorIndex, false);

   el->SetLineColor(m_context.colorManager()->geomColor(kFWTrackerBarrelColorIndex));
   const unsigned int nSegments = 100;
   const double r =  m_context.caloR1();
   for ( unsigned int i = 1; i <= nSegments; ++i )
      el->AddLine(r*sin(TMath::TwoPi()/nSegments*(i-1)), r*cos(TMath::TwoPi()/nSegments*(i-1)), 0,
                  r*sin(TMath::TwoPi()/nSegments*i), r*cos(TMath::TwoPi()/nSegments*i), 0);

   TEvePointSet* ref = new TEvePointSet("reference");
   ref->SetTitle("(0,0,0)");
   ref->SetMarkerStyle(4);
   ref->SetMarkerColor(kWhite);
   ref->SetNextPoint(0.,0.,0.);
   el->AddElement(ref);

   return el;
}

//______________________________________________________________________________

TEveElement*
FWRPZViewGeometry::makeMuonGeometryRhoPhi( void )
{
   Int_t iWheel = 0;
 
   // rho-phi view
   TEveCompound* container = new TEveCompound( "MuonRhoPhi" );


   for( Int_t iStation = 1; iStation <= 4; ++iStation )
   {
      for( Int_t iSector = 1 ; iSector <= 14; ++iSector )
      {
         if( iStation < 4 && iSector > 12 ) continue;
         DTChamberId id( iWheel, iStation, iSector );
	 TEveGeoShape* shape = m_geom->getEveShape( id.rawId() );
	 if( shape ) 
	 {
	    shape->SetMainColor(m_colorComp[kFWMuonBarrelLineColorIndex]->GetMainColor());
	    addToCompound(shape, kFWMuonBarrelLineColorIndex);
	    container->AddElement( shape );
	 }
      }
   }
   return container;
}
namespace {

//void addLibe

}
//______________________________________________________________________________

TEveElement*
FWRPZViewGeometry::makeMuonGeometryRhoZ( void )
{
   TEveElementList* container = new TEveElementList( "MuonRhoZ" );

   {
      TEveCompound* dtContainer = new TEveCompound( "DT" );
      for( Int_t iWheel = -2; iWheel <= 2; ++iWheel )
      {
         for( Int_t iStation = 1; iStation <= 4; ++iStation )
         {
            float min_rho(1000), max_rho(0), min_z(2000), max_z(-2000);

            // This will give us a quarter of DTs
            // which is enough for our projection
            for( Int_t iSector = 1; iSector <= 4; ++iSector )
            {
               DTChamberId id( iWheel, iStation, iSector );
               unsigned int rawid = id.rawId();
               FWGeometry::IdToInfoItr det = m_geom->find( rawid );
               estimateProjectionSizeDT( *det, min_rho, max_rho, min_z, max_z );
            }
            if ( min_rho > max_rho || min_z > max_z ) continue;
            TEveElement* se =  makeShape( min_rho, max_rho, min_z, max_z );
            addToCompound(se, kFWMuonBarrelLineColorIndex);
            dtContainer->AddElement(se);
            se =  makeShape( -max_rho, -min_rho, min_z, max_z );
            addToCompound(se, kFWMuonBarrelLineColorIndex);
            dtContainer->AddElement(se);
         }
      }

      container->AddElement( dtContainer );
   }
   {
      // addcsc
      TEveCompound* cscContainer = new TEveCompound( "CSC" );
      std::vector<CSCDetId> ids;
      for (int endcap = CSCDetId::minEndcapId(); endcap <=  CSCDetId::maxEndcapId(); ++endcap)
      {
         for (int station = 1; station <= 4; ++station)
         {
            ids.push_back(CSCDetId(endcap, station, 2, 10, 0 ));//outer ring up
            ids.push_back(CSCDetId(endcap, station, 2, 11, 0 ));//outer ring up

            ids.push_back(CSCDetId(endcap, station, 2, 28, 0 ));//outer ring down
            ids.push_back(CSCDetId(endcap, station, 2, 29, 0 ));//outer ring down

            ids.push_back(CSCDetId(endcap, station, 1, 5, 0 ));//inner ring up
            ids.push_back(CSCDetId(endcap, station, 1, 6, 0 )); //inner ring up

            int off =  (station == 1) ? 10:0;
            ids.push_back(CSCDetId(endcap, station, 1, 15+off, 0 ));//inner ring down
            ids.push_back(CSCDetId(endcap, station, 1, 16+off, 0 )); //inner ring down
         }
         ids.push_back(CSCDetId(endcap, 1, 3, 10, 0 )); // ring 3 down
         ids.push_back(CSCDetId(endcap, 1, 3, 28, 0 )); // ring 3 down
      }   
      for (std::vector<CSCDetId>::iterator i = ids.begin(); i != ids.end(); ++i)
      {
         unsigned int rawid = i->rawId();
         TEveGeoShape* shape = m_geom->getEveShape(rawid);
         addToCompound(shape, kFWMuonEndcapLineColorIndex);
         shape->SetName(Form(" e:%d r:%d s:%d chamber %d",i->endcap(), i->ring(), i->station(), i->chamber() ));
         cscContainer->AddElement(shape);
      }
      container->AddElement( cscContainer );
   }

   return container;
}

//______________________________________________________________________________

TEveGeoShape*
FWRPZViewGeometry::makeShape( double min_rho, double max_rho, double min_z, double max_z)
{
   TEveTrans t;
   t(1,1) = 1; t(1,2) = 0; t(1,3) = 0;
   t(2,1) = 0; t(2,2) = 1; t(2,3) = 0;
   t(3,1) = 0; t(3,2) = 0; t(3,3) = 1;
   t(1,4) = 0; t(2,4) = (min_rho+max_rho)/2; t(3,4) = (min_z+max_z)/2;

   TEveGeoShape* shape = new TEveGeoShape;
   shape->SetTransMatrix(t.Array());

   shape->SetRnrSelf(kTRUE);
   shape->SetRnrChildren(kTRUE);
   TGeoBBox* box = new TGeoBBox( 0, (max_rho-min_rho)/2, (max_z-min_z)/2 );
   shape->SetShape( box );

   return shape;
}

//______________________________________________________________________________

void
FWRPZViewGeometry::estimateProjectionSizeDT( const FWGeometry::GeomDetInfo& info,
					     float& min_rho, float& max_rho, float& min_z, float& max_z )
{
   // we will test 5 points on both sides ( +/- z)
   float local[3], global[3];

   float dX = info.shape[1];
   float dY = info.shape[2];
   float dZ = info.shape[3];

   local[0] = 0; local[1] = 0; local[2] = dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0] = dX; local[1] = dY; local[2] = dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0] = -dX; local[1] = dY; local[2] = dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0] = dX; local[1] = -dY; local[2] = dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0] = -dX; local[1] = -dY; local[2] = dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0] = 0; local[1] = 0; local[2] = -dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0] = dX; local[1] = dY; local[2] = -dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0] = -dX; local[1] = dY; local[2] = -dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0] = dX; local[1] = -dY; local[2] = -dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0] = -dX; local[1] = -dY; local[2] = -dZ;
   m_geom->localToGlobal( info, local, global );
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );
}


void
FWRPZViewGeometry::estimateProjectionSize( const float* global,
					   float& min_rho, float& max_rho, float& min_z, float& max_z )
{
   double rho = sqrt(global[0] *global[0]+global[1] *global[1]);
   if ( min_rho > rho ) min_rho = rho;
   if ( max_rho < rho ) max_rho = rho;
   if ( min_z > global[2] ) min_z = global[2];
   if ( max_z < global[2] ) max_z = global[2];
}


// ATODO:: check white vertex -> shouldn't be relative to background
//         when detruction ?


// ATODO why color is not set in 3D original, why cast to polygonsetprojected after projected ????
// is geom color dynamic --- independent of projection manager

// NOTE geomtry MuonRhoZAdanced renamed to  MuonRhoZ


//==============================================================================
//==============================================================================



void
FWRPZViewGeometry::showPixelBarrel( bool show )
{
   if( !m_pixelBarrelElements && show )
   { 
      m_pixelBarrelElements = new TEveElementList("PixelBarrel");
      AddElement(m_pixelBarrelElements);
      std::vector<unsigned int> ids = m_geom->getMatchedIds( FWGeometry::Tracker, FWGeometry::PixelBarrel );
      for( std::vector<unsigned int>::const_iterator id = ids.begin();
           id != ids.end(); ++id )
      {
         TEveGeoShape* shape = m_geom->getEveShape( *id );
         shape->SetTitle(Form("PixelBarrel %d",*id));
         addToCompound(shape, kFWPixelBarrelColorIndex);
         m_pixelBarrelElements->AddElement( shape );
      }
      importNew(m_pixelBarrelElements);
   }

   if (m_pixelBarrelElements)
   {
      m_pixelBarrelElements->SetRnrState(show);
      gEve->Redraw3D();
   }
}

void
FWRPZViewGeometry::showPixelEndcap( bool show )
{
   if( !m_pixelEndcapElements && show )
   {
      m_pixelEndcapElements = new TEveElementList("PixelEndcap");

      std::vector<unsigned int> ids = m_geom->getMatchedIds( FWGeometry::Tracker, FWGeometry::PixelEndcap );
      for( std::vector<unsigned int>::const_iterator id = ids.begin();
           id != ids.end(); ++id )
      {
         TEveGeoShape* shape = m_geom->getEveShape( *id );

         shape->SetTitle(Form("PixelEndCap %d",*id));
         addToCompound(shape, kFWPixelEndcapColorIndex);
         m_pixelEndcapElements->AddElement( shape );
      }

      AddElement(m_pixelEndcapElements);
      importNew(m_pixelEndcapElements);
   }

   if (m_pixelEndcapElements)
   {
      m_pixelEndcapElements->SetRnrState(show);
      gEve->Redraw3D();
   }
}


void
FWRPZViewGeometry::showTrackerBarrel( bool show )
{
   if( !m_trackerBarrelElements && show )
   {
      m_trackerBarrelElements = new TEveElementList("TrackerBarrel");

      std::vector<unsigned int> ids = m_geom->getMatchedIds( FWGeometry::Tracker, FWGeometry::TIB );
      for( std::vector<unsigned int>::const_iterator id = ids.begin();
           id != ids.end(); ++id )
      {
         TEveGeoShape* shape = m_geom->getEveShape( *id ); 
         addToCompound(shape, kFWTrackerBarrelColorIndex);
         m_trackerBarrelElements->AddElement( shape );
      }
      ids = m_geom->getMatchedIds( FWGeometry::Tracker, FWGeometry::TOB );
      for( std::vector<unsigned int>::const_iterator id = ids.begin();
           id != ids.end(); ++id )
      {
         TEveGeoShape* shape = m_geom->getEveShape( *id );

         shape->SetTitle(Form("TrackerBarrel %d",*id));
         addToCompound(shape, kFWTrackerBarrelColorIndex);
         m_trackerBarrelElements->AddElement( shape );
      }

      AddElement(m_trackerBarrelElements);
      importNew(m_trackerBarrelElements);
   }

   if (m_trackerBarrelElements)
   {
      m_trackerBarrelElements->SetRnrState(show);
      gEve->Redraw3D();
   }
}

void
FWRPZViewGeometry::showTrackerEndcap( bool show )
{
   if( !m_trackerEndcapElements && show )
   {
      m_trackerEndcapElements = new TEveElementList("TrackerEndcap");

      std::vector<unsigned int> ids = m_geom->getMatchedIds( FWGeometry::Tracker, FWGeometry::TID );
      for( std::vector<unsigned int>::const_iterator id = ids.begin();
           id != ids.end(); ++id )
      {
	 TEveGeoShape* shape = m_geom->getEveShape( *id );
         addToCompound(shape, kFWTrackerEndcapColorIndex);
         m_trackerEndcapElements->AddElement( shape );
      }
      ids = m_geom->getMatchedIds( FWGeometry::Tracker, FWGeometry::TEC );
      for( std::vector<unsigned int>::const_iterator id = ids.begin();
	   id != ids.end(); ++id )
      {
	 TEveGeoShape* shape = m_geom->getEveShape( *id );

         shape->SetTitle(Form("TrackerEndcap %d",*id));
         addToCompound(shape, kFWTrackerEndcapColorIndex);
         m_trackerEndcapElements->AddElement( shape );
      }

      AddElement(m_trackerEndcapElements);
      importNew(m_trackerEndcapElements);
   }

   if (m_trackerEndcapElements)
   {
      m_trackerEndcapElements->SetRnrState(show);
      gEve->Redraw3D();
   }
}

//---------------------------------------------------------
void
FWRPZViewGeometry::showRpcEndcap( bool show )
{
   if( !m_rpcEndcapElements && show )
   {
       m_rpcEndcapElements = new TEveElementList("RpcEndcap");


       std::vector<RPCDetId> ids;
       int mxSt = m_geom->versionInfo().haveExtraDet("RE4") ? 4:3; 
       for (int region = -1; region <=1; ++ region )
       {
           if (region == 0 ) continue;
           for (int ring = 2; ring <= 3; ++ring) {
             for (int station = 1; station <= mxSt; ++station ) {
                   int sector = 1;
                   ids.push_back(RPCDetId(region, ring, station, sector, 1, 1, 1));
                   ids.push_back(RPCDetId(region, ring, station, sector, 1, 1, 2));
                   ids.push_back(RPCDetId(region, ring, station, sector, 1, 1, 3));
                   if (ring == 2 && station == 1) { // 2 layers in ring 2 station 1 up 
                       ids.push_back(RPCDetId(region, ring, station, sector, 1, 2, 1));
                       ids.push_back(RPCDetId(region, ring, station, sector, 1, 2, 2));
                       ids.push_back(RPCDetId(region, ring, station, sector, 1, 2, 3));
                   }
                   sector = 5;
                   ids.push_back(RPCDetId(region, ring, station, sector, 1, 1, 1));
                   ids.push_back(RPCDetId(region, ring, station, sector, 1, 1, 2));
                   ids.push_back(RPCDetId(region, ring, station, sector, 1, 1, 3));

                   if (ring == 2 && station == 1) { // 2 layers in ring 2 station 1 down
                       ids.push_back(RPCDetId(region, ring, station, sector, 1, 2, 1));
                       ids.push_back(RPCDetId(region, ring, station, sector, 1, 2, 2));
                       ids.push_back(RPCDetId(region, ring, station, sector, 1, 2, 3));
                   }
               }
           }
       }

      for (std::vector<RPCDetId>::iterator i = ids.begin(); i != ids.end(); ++i)
      {
         TEveGeoShape* shape = m_geom->getEveShape(i->rawId());
         addToCompound(shape, kFWMuonEndcapLineColorIndex);
         m_rpcEndcapElements->AddElement(shape);
         gEve->AddToListTree(shape, true);
      }
   
      AddElement(m_rpcEndcapElements);
      importNew(m_rpcEndcapElements);
   }

   if (m_rpcEndcapElements)
   {
      m_rpcEndcapElements->SetRnrState(show);
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________

void
FWRPZViewGeometry::showGEM( bool show )
{
  // hardcoded gem and me0; need to find better way for different gem geometries
  if( !m_GEMElements && show ){
    m_GEMElements = new TEveElementList("GEM");

    for( Int_t iRegion = GEMDetId::minRegionId; iRegion <= GEMDetId::maxRegionId; iRegion= iRegion+2){
      int mxSt = m_geom->versionInfo().haveExtraDet("GE2") ? 3:1; 

      for( Int_t iStation = GEMDetId::minStationId; iStation <= mxSt; ++iStation ){	      
	Int_t iRing = 1;
	for( Int_t iLayer = GEMDetId::minLayerId; iLayer <= GEMDetId::maxLayerId ; ++iLayer ){
	  int maxChamber = 36;
	  if (iStation >= 2) maxChamber = 18;

	  for( Int_t iChamber = 1; iChamber <= maxChamber; ++iChamber ){
	    int maxRoll = iChamber%2 ? 9:10;
	    if (iStation == 2) maxRoll = 8;
	    if (iStation == 3) maxRoll = 12;

	    for (Int_t iRoll = GEMDetId::minRollId; iRoll <= maxRoll ; ++iRoll ){
	      GEMDetId id( iRegion, iRing, iStation, iLayer, iChamber, iRoll );
	      TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
	      if (shape){
		addToCompound(shape, kFWMuonEndcapLineColorIndex);
		m_GEMElements->AddElement( shape );
		gEve->AddToListTree(shape, true);
	      }
	    }
	  }
	}
      }
    }
      
    AddElement(m_GEMElements);
    importNew(m_GEMElements);
  }
  if (m_GEMElements){
    m_GEMElements->SetRnrState(show);
    gEve->Redraw3D();
  }
}

void
FWRPZViewGeometry::showME0( bool show )
{
  if( !m_ME0Elements && show ){
    m_ME0Elements = new TEveElementList("ME0");

    for( Int_t iRegion = ME0DetId::minRegionId; iRegion <= ME0DetId::maxRegionId; iRegion= iRegion+2 ){
      for( Int_t iLayer = 1; iLayer <= 6 ; ++iLayer ){
	for( Int_t iChamber = 1; iChamber <= 18; ++iChamber ){
	  Int_t iRoll = 1;
	  ME0DetId id( iRegion, iLayer, iChamber, iRoll );
	  TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
	  if (shape){
	    addToCompound(shape, kFWMuonEndcapLineColorIndex);
	    m_ME0Elements->AddElement( shape );
	    gEve->AddToListTree(shape, true);
	  }
	}
      }
    }
      
    AddElement(m_ME0Elements);
    importNew(m_ME0Elements);
  }
  if (m_ME0Elements){
    m_ME0Elements->SetRnrState(show);
    gEve->Redraw3D();
  }
}

//-------------------------------------

void FWRPZViewGeometry::importNew(TEveElementList* x)
{
   TEveProjected* proj =  *BeginProjecteds();
   proj->GetManager()->SubImportElements(x,  proj->GetProjectedAsElement());
                               
}
