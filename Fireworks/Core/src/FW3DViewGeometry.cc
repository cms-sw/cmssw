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
// $Id: FW3DViewGeometry.cc,v 1.1 2010/04/06 20:00:35 amraktad Exp $
//

// system include files
#include <sstream>

// user include files

#include "TEveManager.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FW3DViewGeometry.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TEveElementIter.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DViewGeometry::FW3DViewGeometry(const DetIdToMatrix* m):
   TEveElementList("3DViewGeo"),
   m_detIdToMatrix(m),
   m_muonBarrelElements(0),
   m_muonEndcapElements(0),
   m_pixelBarrelElements(0),
   m_pixelEndcapElements(0),
   m_trackerBarrelElements(0),
   m_trackerEndcapElements(0),

   m_geomTransparency(95)
{
}

// FW3DViewGeometry::FW3DViewGeometry(const FW3DViewGeometry& rhs)
// {
//    // do actual copying here;
// }

FW3DViewGeometry::~FW3DViewGeometry()
{
}

//
// assignment operators
//
// const FW3DViewGeometry& FW3DViewGeometry::operator=(const FW3DViewGeometry& rhs)
// {
//   //An exception safe implementation is
//   FW3DViewGeometry temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

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
FW3DViewGeometry::showMuonBarrel(bool  showMuonBarrel)
{
   if (!m_muonBarrelElements && showMuonBarrel)
   {
      m_muonBarrelElements = new TEveElementList( "DT" );
      for ( Int_t iWheel = -2; iWheel <= 2; ++iWheel)
      {
         for (Int_t iStation = 1; iStation <= 4; ++iStation)
         {
            std::ostringstream s;
            s << "Station" << iStation;
            TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
            m_muonBarrelElements->AddElement( cStation );
            for (Int_t iSector = 1 ; iSector <= 14; ++iSector)
            {
               if ( iStation < 4 && iSector > 12 ) continue;
               DTChamberId id(iWheel, iStation, iSector);
               TEveGeoShape* shape = m_detIdToMatrix->getShape( id.rawId() );
               if ( !shape ) continue;
               shape->SetMainTransparency(m_geomTransparency);
               cStation->AddElement(shape);
            }
         }
      }
      AddElement(m_muonBarrelElements);
   }

   if (m_muonBarrelElements)
   {
      m_muonBarrelElements->SetRnrState(showMuonBarrel);
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showMuonEndcap(bool showMuonEndcap )
{
   if ( showMuonEndcap && !m_muonEndcapElements )
   {
      m_muonEndcapElements = new TEveElementList( "CSC" );
      for ( Int_t iEndcap = 1; iEndcap <= 2; ++iEndcap ) { // 1=forward (+Z), 2=backward(-Z)
         TEveElementList* cEndcap = 0;
         if (iEndcap == 1)
            cEndcap = new TEveElementList( "Forward" );
         else
            cEndcap = new TEveElementList( "Backward" );
         m_muonEndcapElements->AddElement( cEndcap );
         for ( Int_t iStation=1; iStation<=4; ++iStation)
         {
            std::ostringstream s; s << "Station" << iStation;
            TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
            cEndcap->AddElement( cStation );
            for ( Int_t iRing=1; iRing<=4; ++iRing) {
               if (iStation > 1 && iRing > 2) continue;
               std::ostringstream s; s << "Ring" << iRing;
               TEveElementList* cRing  = new TEveElementList( s.str().c_str() );
               cStation->AddElement( cRing );
               for ( Int_t iChamber=1; iChamber<=72; ++iChamber)
               {
                  if (iStation>1 && iChamber>36) continue;
                  Int_t iLayer = 0; // chamber
                  // exception is thrown if parameters are not correct and I keep
                  // forgetting how many chambers we have in each ring.
                  try {
                     CSCDetId id(iEndcap, iStation, iRing, iChamber, iLayer);
                     TEveGeoShape* shape = m_detIdToMatrix->getShape( id.rawId() );
                     if ( !shape ) continue;
                     shape->SetMainTransparency(m_geomTransparency);
                     cRing->AddElement( shape );
                  }
                  catch (... ) {}
               }
            }
         }
      }
      AddElement( m_muonEndcapElements);
   }

   if (m_muonEndcapElements)
   {
      m_muonEndcapElements->SetRnrState(showMuonEndcap);
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showPixelBarrel(bool showPixelBarrel )
{
   if ( showPixelBarrel && !m_pixelBarrelElements )
   {
      m_pixelBarrelElements = new TEveElementList( "PixelBarrel" );
      m_pixelBarrelElements->SetRnrState(showPixelBarrel);
      std::vector<unsigned int> ids = m_detIdToMatrix->getMatchedIds("PixelBarrel");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_detIdToMatrix->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency);
         m_pixelBarrelElements->AddElement( shape );
      }
      AddElement( m_pixelBarrelElements);
   }

   if (m_pixelBarrelElements)
   {
      m_pixelBarrelElements->SetRnrState(showPixelBarrel);
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showPixelEndcap(bool  showPixelEndcap )
{
   if ( showPixelEndcap && ! m_pixelEndcapElements )
   {
      m_pixelEndcapElements = new TEveElementList( "PixelEndcap" );
      std::vector<unsigned int> ids = m_detIdToMatrix->getMatchedIds("PixelForward");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_detIdToMatrix->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency);
         m_pixelEndcapElements->AddElement( shape );
      }
      AddElement(m_pixelEndcapElements);
   }

   if (m_pixelEndcapElements)
   {
      m_pixelEndcapElements->SetRnrState(showPixelEndcap);
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showTrackerBarrel(bool  showTrackerBarrel )
{
   if (  showTrackerBarrel &&  !m_trackerBarrelElements )
   {
      m_trackerBarrelElements = new TEveElementList( "TrackerBarrel" );
      m_trackerBarrelElements->SetRnrState(showTrackerBarrel);
      std::vector<unsigned int> ids = m_detIdToMatrix->getMatchedIds("tib:TIB");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_detIdToMatrix->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency);
         m_trackerBarrelElements->AddElement( shape );
      }
      ids = m_detIdToMatrix->getMatchedIds("tob:TOB");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_detIdToMatrix->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency);
         m_trackerBarrelElements->AddElement( shape );
      }
      AddElement(m_trackerBarrelElements);
   }

   if (m_trackerBarrelElements )
   {
      m_trackerBarrelElements->SetRnrState(showTrackerBarrel);
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void
FW3DViewGeometry::showTrackerEndcap(bool showTrackerEndcap )
{
   if ( showTrackerEndcap && !m_trackerEndcapElements )
   {
      m_trackerEndcapElements = new TEveElementList( "TrackerEndcap" );
      std::vector<unsigned int> ids = m_detIdToMatrix->getMatchedIds("tid:TID");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_detIdToMatrix->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency);
         m_trackerEndcapElements->AddElement( shape );
      }
      ids = m_detIdToMatrix->getMatchedIds("tec:TEC");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_detIdToMatrix->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency);
         m_trackerEndcapElements->AddElement( shape );
      }
     AddElement( m_trackerEndcapElements);
   }

   if (m_trackerEndcapElements )
   {
      m_trackerEndcapElements->SetRnrState(showTrackerEndcap);
      gEve->Redraw3D();
   }
}

void
FW3DViewGeometry::setTransparency(int transp )
{
   m_geomTransparency = transp;
   if ( m_muonBarrelElements ) {
      TEveElementIter iter(m_muonBarrelElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(transp);
         iter.next();
      }
   }
   if ( m_muonEndcapElements ) {
      TEveElementIter iter(m_muonEndcapElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(transp);
         iter.next();
      }
   }
   if ( m_pixelBarrelElements ) {
      TEveElementIter iter(m_pixelBarrelElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(transp);
         iter.next();
      }
   }
   if ( m_pixelEndcapElements ) {
      TEveElementIter iter(m_pixelEndcapElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(transp);
         iter.next();
      }
   }
   if ( m_trackerBarrelElements ) {
      TEveElementIter iter(m_trackerBarrelElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(transp);
         iter.next();
      }
   }
   if ( m_trackerEndcapElements ) {
      TEveElementIter iter(m_trackerEndcapElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(transp);
         iter.next();
      }
   }
}
