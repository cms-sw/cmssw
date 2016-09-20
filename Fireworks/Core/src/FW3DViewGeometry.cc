// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DViewGeometry
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Thu Mar 25 22:06:57 CET 2010
//

// system include files
#include <sstream>

// user include files

#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FW3DViewGeometry.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWColorManager.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DViewGeometry::FW3DViewGeometry(const fireworks::Context& context):
   FWViewGeometryList(context, false),
   m_muonBarrelElements(0), m_muonBarrelFullElements(0),
   m_muonEndcapElements(0), m_muonEndcapFullElements(0),
   m_pixelBarrelElements(0),
   m_pixelEndcapElements(0),
   m_trackerBarrelElements(0),
   m_trackerEndcapElements(0)
{  

   SetElementName("3D Geometry");
}

// FW3DViewGeometry::FW3DViewGeometry(const FW3DViewGeometry& rhs)
// {
//    // do actual copying here;
// }

FW3DViewGeometry::~FW3DViewGeometry()
{
}


//
// member functions
//

//
// const member functions
//

//
// static member functions
//

void
FW3DViewGeometry::showMuonBarrel( bool showMuonBarrel )
{
   if( !m_muonBarrelElements && showMuonBarrel )
   {
      m_muonBarrelElements = new TEveElementList( "DT" );
      for( Int_t iWheel = -2; iWheel <= 2; ++iWheel )
      {
         for ( Int_t iStation = 1; iStation <= 4; ++iStation )
         {
	    // We display only the outer chambers to make the event look more
	    // prominent
	    if( iWheel == -2 || iWheel == 2 || iStation == 4 )
	    {
	       std::ostringstream s;
	       s << "Station" << iStation;
	       TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
	       m_muonBarrelElements->AddElement( cStation );
	       for( Int_t iSector = 1 ; iSector <= 14; ++iSector )
	       {
		  if( iStation < 4 && iSector > 12 ) continue;
		  DTChamberId id( iWheel, iStation, iSector );
		  TEveGeoShape* shape = m_geom->getEveShape( id.rawId() );
                  addToCompound(shape, kFWMuonBarrelLineColorIndex);
		  cStation->AddElement( shape );
	       }
	    }
         }
      }
      AddElement( m_muonBarrelElements );
   }

   if( m_muonBarrelElements )
   {
      m_muonBarrelElements->SetRnrState( showMuonBarrel );
      gEve->Redraw3D();
   }
}

void
FW3DViewGeometry::showMuonBarrelFull(bool showMuonBarrel)
{
   if (!m_muonBarrelFullElements && showMuonBarrel)
   {
      m_muonBarrelFullElements = new TEveElementList( "DT Full" );
      for (Int_t iWheel = -2; iWheel <= 2; ++iWheel)
      {
         TEveElementList* cWheel = new TEveElementList(TString::Format("Wheel %d", iWheel));
         m_muonBarrelFullElements->AddElement(cWheel);
         for (Int_t iStation = 1; iStation <= 4; ++iStation)
         {
            TEveElementList* cStation  = new TEveElementList(TString::Format("Station %d", iStation));
            cWheel->AddElement(cStation);
            for (Int_t iSector = 1 ; iSector <= 14; ++iSector)
            {
               if( iStation < 4 && iSector > 12 ) continue;
               DTChamberId id( iWheel, iStation, iSector );
               TEveGeoShape* shape = m_geom->getEveShape(id.rawId());
               shape->SetTitle(TString::Format("DT: W=%d, S=%d, Sec=%d\ndet-id=%u",
                                               iWheel, iStation, iSector, id.rawId()));
               addToCompound(shape, kFWMuonBarrelLineColorIndex);
               cStation->AddElement(shape);
            }
         }
      }
      AddElement(m_muonBarrelFullElements);
   }

   if (m_muonBarrelFullElements)
   {
      m_muonBarrelFullElements->SetRnrState(showMuonBarrel);
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showMuonEndcap( bool showMuonEndcap )
{
  if( showMuonEndcap && !m_muonEndcapElements )
   {
      m_muonEndcapElements = new TEveElementList( "EndCap" );

      for( Int_t iEndcap = 1; iEndcap <= 2; ++iEndcap ) // 1=forward (+Z), 2=backward(-Z)
      { 
         TEveElementList* cEndcap = 0;
         if( iEndcap == 1 )
            cEndcap = new TEveElementList( "CSC Forward" );
         else
            cEndcap = new TEveElementList( "CSC Backward" );
         m_muonEndcapElements->AddElement( cEndcap );
      	 // Actual CSC geometry:
      	 // Station 1 has 4 rings with 36 chambers in each
      	 // Station 2: ring 1 has 18 chambers, ring 2 has 36 chambers
      	 // Station 3: ring 1 has 18 chambers, ring 2 has 36 chambers
      	 // Station 4: ring 1 has 18 chambers
      	 Int_t maxChambers = 36;
         for( Int_t iStation = 1; iStation <= 4; ++iStation )
         {
            std::ostringstream s; s << "Station" << iStation;
            TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
            cEndcap->AddElement( cStation );
            for( Int_t iRing = 1; iRing <= 4; ++iRing )
      	    {
               if( iStation > 1 && iRing > 2 ) continue;
               // if( iStation > 3 && iRing > 1 ) continue;
               std::ostringstream s; s << "Ring" << iRing;
               TEveElementList* cRing  = new TEveElementList( s.str().c_str() );
               cStation->AddElement( cRing );
      	       ( iRing == 1 && iStation > 1 ) ? ( maxChambers = 18 ) : ( maxChambers = 36 );
               for( Int_t iChamber = 1; iChamber <= maxChambers; ++iChamber )
               {
                  Int_t iLayer = 0; // chamber
      		  CSCDetId id( iEndcap, iStation, iRing, iChamber, iLayer );
      		  TEveGeoShape* shape = m_geom->getEveShape( id.rawId() );
                  shape->SetTitle(TString::Format("CSC: %s, S=%d, R=%d, C=%d\ndet-id=%u",
                                                  cEndcap->GetName(), iStation, iRing, iChamber, id.rawId()));
 	  	            
                  addToCompound(shape, kFWMuonEndcapLineColorIndex);
      		  cRing->AddElement( shape );
               }
            }
      	 }
      }
      // hardcoded gem and me0; need to find better way for different gem geometries
      for( Int_t iRegion = GEMDetId::minRegionId; iRegion <= GEMDetId::maxRegionId; iRegion= iRegion+2){
	TEveElementList* teEndcap = 0;
	if( iRegion == 1 )
	  teEndcap = new TEveElementList( "GEM Forward" );
	else
	  teEndcap = new TEveElementList( "GEM Backward" );
	m_muonEndcapElements->AddElement( teEndcap );

	int mxSt = m_geom->versionInfo().haveExtraDet("GE2") ? 3:1; 

	for( Int_t iStation = GEMDetId::minStationId; iStation <= mxSt; ++iStation ){
	  std::ostringstream s; s << "Station" << iStation;
	  TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
	  teEndcap->AddElement( cStation );
	      
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
		TEveGeoShape* shape = m_geom->getEveShape( id.rawId() );
		if (shape){
		  shape->SetTitle(TString::Format("GEM: , Rng=%d, St=%d, Ch=%d Rl=%d\ndet-id=%u",
						  iRing, iStation, iChamber, iRoll, id.rawId()));
 	  	            
		  cStation->AddElement( shape );
		  addToCompound(shape, kFWMuonEndcapLineColorIndex);
		}
	      }
	    }
	  }
	}
      }

      // adding me0
      if (m_geom->versionInfo().haveExtraDet("ME0") ){
	for( Int_t iRegion = ME0DetId::minRegionId; iRegion <= ME0DetId::maxRegionId; iRegion= iRegion+2 ){
	  TEveElementList* teEndcap = 0;
	  if( iRegion == 1 )
	    teEndcap = new TEveElementList( "ME0 Forward" );
	  else
	    teEndcap = new TEveElementList( "ME0 Backward" );
	  m_muonEndcapElements->AddElement( teEndcap );

	  for( Int_t iLayer = 1; iLayer <= 6 ; ++iLayer ){
	    std::ostringstream s; s << "Layer" << iLayer;
	    TEveElementList* cLayer  = new TEveElementList( s.str().c_str() );
	    teEndcap->AddElement( cLayer );

	    for( Int_t iChamber = 1; iChamber <= 18; ++iChamber ){
	      Int_t iRoll = 1;
	      // for (Int_t iRoll = ME0DetId::minRollId; iRoll <= ME0DetId::maxRollId ; ++iRoll ){
	      ME0DetId id( iRegion, iLayer, iChamber, iRoll );
	      TEveGeoShape* shape = m_geom->getEveShape( id.rawId() );
	      if (shape){
		shape->SetTitle(TString::Format("ME0: , Ch=%d Rl=%d\ndet-id=%u",
						iChamber, iRoll, id.rawId()));
 	  	            
		addToCompound(shape, kFWMuonEndcapLineColorIndex );
		cLayer->AddElement( shape );
	      }
	    }
	  }
	}
      }
      
      AddElement( m_muonEndcapElements );
   }

   if( m_muonEndcapElements )
   {
      m_muonEndcapElements->SetRnrState( showMuonEndcap );
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showPixelBarrel( bool showPixelBarrel )
{
   if( showPixelBarrel && !m_pixelBarrelElements )
   {
      m_pixelBarrelElements = new TEveElementList( "PixelBarrel" );
      m_pixelBarrelElements->SetRnrState( showPixelBarrel );
      std::vector<unsigned int> ids = m_geom->getMatchedIds( FWGeometry::Tracker, FWGeometry::PixelBarrel );
      for( std::vector<unsigned int>::const_iterator id = ids.begin();
	   id != ids.end(); ++id )
      {
	 TEveGeoShape* shape = m_geom->getEveShape( *id );
	 PXBDetId idid = PXBDetId( *id );
	 unsigned int layer = idid.layer();
	 unsigned int ladder = idid.ladder();
	 unsigned int module = idid.module();
	 
         shape->SetTitle( TString::Format( "PixelBarrel %d: Layer=%u, Ladder=%u, Module=%u",
					   *id, layer, ladder, module ));

         addToCompound(shape, kFWPixelBarrelColorIndex);
         m_pixelBarrelElements->AddElement( shape );
      }
      AddElement( m_pixelBarrelElements );
   }

   if( m_pixelBarrelElements )
   {
      m_pixelBarrelElements->SetRnrState( showPixelBarrel );
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showPixelEndcap(bool  showPixelEndcap )
{
   if( showPixelEndcap && ! m_pixelEndcapElements )
   {
      m_pixelEndcapElements = new TEveElementList( "PixelEndcap" );
      std::vector<unsigned int> ids = m_geom->getMatchedIds( FWGeometry::Tracker, FWGeometry::PixelEndcap );
      for( std::vector<unsigned int>::const_iterator id = ids.begin();
	   id != ids.end(); ++id )
      {
	 TEveGeoShape* shape = m_geom->getEveShape( *id );
	 PXFDetId idid = PXFDetId( *id );
	 unsigned int side = idid.side();
	 unsigned int disk = idid.disk();
	 unsigned int blade = idid.blade();
	 unsigned int panel = idid.panel();
	 unsigned int module = idid.module();

         shape->SetTitle( TString::Format( "PixelEndcap %d: Side=%u, Disk=%u, Blade=%u, Panel=%u, Module=%u",
					   *id, side, disk, blade, panel, module ));
	 
         addToCompound(shape, kFWPixelEndcapColorIndex);
         m_pixelEndcapElements->AddElement( shape );
      }
      AddElement( m_pixelEndcapElements );
   }

   if( m_pixelEndcapElements )
   {
      m_pixelEndcapElements->SetRnrState( showPixelEndcap );
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showTrackerBarrel( bool  showTrackerBarrel )
{
   if( showTrackerBarrel && ! m_trackerBarrelElements )
   {
      m_trackerBarrelElements = new TEveElementList( "TrackerBarrel" );
      m_trackerBarrelElements->SetRnrState( showTrackerBarrel );
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
      AddElement( m_trackerBarrelElements );
   }

   if( m_trackerBarrelElements )
   {
      m_trackerBarrelElements->SetRnrState( showTrackerBarrel );
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showTrackerEndcap( bool showTrackerEndcap )
{
   if( showTrackerEndcap && ! m_trackerEndcapElements )
   {
      m_trackerEndcapElements = new TEveElementList( "TrackerEndcap" );
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
      AddElement( m_trackerEndcapElements );
   }

   if (m_trackerEndcapElements )
   {
      m_trackerEndcapElements->SetRnrState( showTrackerEndcap );
      gEve->Redraw3D();
   }
}

 
