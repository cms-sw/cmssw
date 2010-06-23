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
// $Id: FWRPZViewGeometry.cc,v 1.3 2010/06/18 10:17:16 yana Exp $
//

// system include files
#include <iostream>
#include <sstream>

// user include files

#include "TClass.h"
#include "TGeoManager.h"
#include "TGeoBBox.h"

#include "TEveElement.h"
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"
#include "TEvePolygonSetProjected.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FWRPZViewGeometry.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

//
//
//
// constants, enums and typedefs
//

//
// static data member definitions
//
TEveElementList* FWRPZViewGeometry::s_rhoPhiGeo = 0;
TEveElementList* FWRPZViewGeometry::s_rhoZGeo   = 0;

//
// constructors and destructor
//
FWRPZViewGeometry::FWRPZViewGeometry(const DetIdToMatrix* m,  const FWColorManager* colMng):
   m_detIdToMatrix(m),
   m_colorManager(colMng)
{
}

// FWRPZViewGeometry::FWRPZViewGeometry(const FWRPZViewGeometry& rhs)
// {
//    // do actual copying here;
// }

FWRPZViewGeometry::~FWRPZViewGeometry()
{
   s_rhoPhiGeo->DecDenyDestroy();
   s_rhoZGeo->DecDenyDestroy();
}

//______________________________________________________________________________

TEveElement*
FWRPZViewGeometry::getGeoElements(const FWViewType::EType type)
{
   if (type == FWViewType::kRhoPhi)
   {
      if ( !s_rhoPhiGeo)
      {
         s_rhoPhiGeo = new TEveElementList("Geomtery RhoPhi");
         s_rhoPhiGeo->IncDenyDestroy();
         s_rhoPhiGeo->AddElement(makeMuonGeometryRhoPhi());
         s_rhoPhiGeo->AddElement(makeTrackerGeometryRhoPhi());

      }
      return s_rhoPhiGeo;
   }
   else 
   {
      if ( !s_rhoZGeo)
      {
         s_rhoZGeo = new TEveElementList("Geomtery RhoZ");
         
         s_rhoZGeo->IncDenyDestroy();
         s_rhoZGeo->AddElement(makeMuonGeometryRhoZ());
         s_rhoZGeo->AddElement(makeTrackerGeometryRhoZ());
      }
      return s_rhoZGeo;
   }
   return 0;
}

//______________________________________________________________________________


TEveElement*
FWRPZViewGeometry::makeTrackerGeometryRhoZ()
{
   TEveElementList* list = new TEveElementList( "TrackerRhoZ" );

   TEvePointSet* ref = new TEvePointSet("reference");
   ref->SetPickable(kTRUE);
   ref->SetTitle("(0,0,0)");
   ref->SetMarkerStyle(4);
   ref->SetMarkerColor(kWhite);
   ref->SetNextPoint(0.,0.,0.);
   list->AddElement(ref);

   TEveStraightLineSet* el = new TEveStraightLineSet( "outline" );
   el->SetPickable(kFALSE);
   el->SetLineColor(m_colorManager->geomColor(kFWTrackerColorIndex));
   el->AddLine(0, 123,-300, 0, 123, 300);
   el->AddLine(0, 123, 300, 0,-123, 300);
   el->AddLine(0,-123, 300, 0,-123,-300);
   el->AddLine(0,-123,-300, 0, 123,-300);
   list->AddElement(el);
 
   return el;

}

TEveElement*
FWRPZViewGeometry::makeTrackerGeometryRhoPhi()
{
   TEveStraightLineSet* el = new TEveStraightLineSet( "TrackerRhoPhi" );
   el->SetLineColor(m_colorManager->geomColor(kFWTrackerColorIndex));
   const unsigned int nSegments = 100;
   const double r = 123;
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
FWRPZViewGeometry::makeMuonGeometryRhoPhi()
{
   // rho-phi view
   TEveElementList* container = new TEveElementList( "MuonRhoPhi" );
   Int_t iWheel = 0;
   for (Int_t iStation = 1; iStation <= 4; ++iStation)
   {
      std::ostringstream s;
      s << "Station" << iStation;
      TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
      container->AddElement( cStation );
      for (Int_t iSector = 1 ; iSector <= 14; ++iSector)
      {
         if ( iStation < 4 && iSector > 12 ) continue;
         DTChamberId id(iWheel, iStation, iSector);
         TEveGeoShape* shape = m_detIdToMatrix->getShape( id.rawId() );
         if ( shape ) cStation->AddElement(shape);
      }
      
      // set background geometry visibility parameters
      TEveElementIter iter(container);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(50);
         element->SetMainColor(m_colorManager->geomColor(kFWMuonBarrelMainColorIndex));
         if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
            poly->SetLineColor(m_colorManager->geomColor(kFWMuonBarrelLineColorIndex));
         iter.next();
      }
      
   }

   return container;
}

//______________________________________________________________________________

TEveElement*
FWRPZViewGeometry::makeMuonGeometryRhoZ()
{
   // lets project everything by hand
   if ( !m_detIdToMatrix ) return 0;

   TEveElementList* container = new TEveElementList( "MuonRhoZ" );

   TEveElementList* dtContainer = new TEveElementList( "DT" );
   container->AddElement( dtContainer );

   for ( Int_t iWheel = -2; iWheel <= 2; ++iWheel ) {
      std::ostringstream s; s << "Wheel" << iWheel;
      TEveElementList* cWheel  = new TEveElementList( s.str().c_str() );
      dtContainer->AddElement( cWheel );
      for ( Int_t iStation=1; iStation<=4; ++iStation) {
         std::ostringstream s; s << "Station" << iStation;
         double min_rho(1000), max_rho(0), min_z(2000), max_z(-2000);

         for ( Int_t iSector=1; iSector<=14; ++iSector) {
            if (iStation<4 && iSector>12) continue;
            DTChamberId id(iWheel, iStation, iSector);
            TEveGeoShape* shape = m_detIdToMatrix->getShape( id.rawId() );
            if (!shape ) continue;
            estimateProjectionSizeDT( m_detIdToMatrix->getMatrix( id.rawId() ),
                                      shape->GetShape(), min_rho, max_rho, min_z, max_z );
         }
         if ( min_rho > max_rho || min_z > max_z ) continue;
         cWheel->AddElement( makeShape( s.str().c_str(), min_rho, max_rho, min_z, max_z ) );
         cWheel->AddElement( makeShape( s.str().c_str(), -max_rho, -min_rho, min_z, max_z ) );
      }
   }

   TEveElementList* cscContainer = new TEveElementList( "CSC" );
   container->AddElement( cscContainer );
   for ( Int_t iEndcap = 1; iEndcap <= 2; ++iEndcap ) { // 1=forward (+Z), 2=backward(-Z)
      TEveElementList* cEndcap = 0;
      if (iEndcap == 1)
         cEndcap = new TEveElementList( "Forward" );
      else
         cEndcap = new TEveElementList( "Backward" );
      cscContainer->AddElement( cEndcap );
      // Actual CSC geometry:
      // Station 1 has 4 rings with 36 chambers in each
      // Station 2: ring 1 has 18 chambers, ring 2 has 36 chambers
      // Station 3: ring 1 has 18 chambers, ring 2 has 36 chambers
      // Station 4: ring 1 has 18 chambers
      Int_t maxChambers = 36;
      for ( Int_t iStation=1; iStation<=4; ++iStation) {
         std::ostringstream s; s << "Station" << iStation;
         TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
         cEndcap->AddElement( cStation );
         for ( Int_t iRing=1; iRing<=4; ++iRing) {
            if (iStation > 1 && iRing > 2) continue;
	    if( iStation > 3 && iRing > 1 ) continue;
            std::ostringstream s; s << "Ring" << iRing;
            double min_rho(1000), max_rho(0), min_z(2000), max_z(-2000);
	    ( iRing == 1 && iStation > 1 ) ? ( maxChambers = 18 ) : ( maxChambers = 36 );
            for ( Int_t iChamber=1; iChamber<=maxChambers; ++iChamber) {
               Int_t iLayer = 0; // chamber
	       CSCDetId id(iEndcap, iStation, iRing, iChamber, iLayer);
	       TEveGeoShape* shape = m_detIdToMatrix->getShape( id.rawId() );
	       if ( !shape ) continue;
	       // get matrix from the main geometry, since detIdToGeo has reflected and
	       // rotated reference frame to get proper local coordinates
	       TGeoManager* manager = m_detIdToMatrix->getManager();
	       manager->cd( m_detIdToMatrix->getPath( id.rawId() ) );
	       const TGeoHMatrix* matrix = manager->GetCurrentMatrix();
	       estimateProjectionSizeCSC( matrix, shape->GetShape(), min_rho, max_rho, min_z, max_z );
            }
            if ( min_rho > max_rho || min_z > max_z ) continue;
            cStation->AddElement( makeShape( s.str().c_str(), min_rho, max_rho, min_z, max_z ) );
            cStation->AddElement( makeShape( s.str().c_str(), -max_rho, -min_rho, min_z, max_z ) );
         }
      }
   }
   
   {
      TEveElementIter iter(dtContainer);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(50);
         element->SetMainColor(m_colorManager->geomColor(kFWMuonBarrelMainColorIndex));
         if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
            poly->SetLineColor(m_colorManager->geomColor(kFWMuonBarrelLineColorIndex));
         iter.next();
      }
   }
   
   {
      TEveElementIter iter(cscContainer);
      while ( iter.current() ) {
         iter.current()->SetMainTransparency(50);
         iter.current()->SetMainColor(m_colorManager->geomColor(kFWMuonEndCapMainColorIndex));
         if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(iter.current()) )
            poly->SetLineColor(m_colorManager->geomColor(kFWMuonEndCapLineColorIndex));
         iter.next();
      }
   }
   
   return container;
}

//______________________________________________________________________________

TEveGeoShape*
FWRPZViewGeometry::makeShape( const char* name,
                              double min_rho, double max_rho, double min_z, double max_z )
{
   TEveTrans t;
   t(1,1) = 1; t(1,2) = 0; t(1,3) = 0;
   t(2,1) = 0; t(2,2) = 1; t(2,3) = 0;
   t(3,1) = 0; t(3,2) = 0; t(3,3) = 1;
   t(1,4) = 0; t(2,4) = (min_rho+max_rho)/2; t(3,4) = (min_z+max_z)/2;

   TEveGeoShape* shape = new TEveGeoShape(name);
   shape->SetTransMatrix(t.Array());

   shape->SetRnrSelf(kTRUE);
   shape->SetRnrChildren(kTRUE);
   TGeoBBox* box = new TGeoBBox( 0, (max_rho-min_rho)/2, (max_z-min_z)/2 );
   shape->SetShape( box );
   return shape;
}

//______________________________________________________________________________

void FWRPZViewGeometry::estimateProjectionSizeDT( const TGeoHMatrix* matrix, const TGeoShape* shape,
                                                     double& min_rho, double& max_rho, double& min_z, double& max_z )
{
   const TGeoBBox* box = dynamic_cast<const TGeoBBox*>( shape );
   if ( !box ) return;

   // we will test 5 points on both sides ( +/- z)
   Double_t local[3], global[3];

   local[0]=0; local[1]=0; local[2]=box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=box->GetDX(); local[1]=box->GetDY(); local[2]=box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=-box->GetDX(); local[1]=box->GetDY(); local[2]=box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=box->GetDX(); local[1]=-box->GetDY(); local[2]=box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=-box->GetDX(); local[1]=-box->GetDY(); local[2]=box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=0; local[1]=0; local[2]=-box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=box->GetDX(); local[1]=box->GetDY(); local[2]=-box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=-box->GetDX(); local[1]=box->GetDY(); local[2]=-box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=box->GetDX(); local[1]=-box->GetDY(); local[2]=-box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=-box->GetDX(); local[1]=-box->GetDY(); local[2]=-box->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );
}

void FWRPZViewGeometry::estimateProjectionSizeCSC( const TGeoHMatrix* matrix, const TGeoShape* shape,
                                                      double& min_rho, double& max_rho, double& min_z, double& max_z )
{
   // const TGeoTrap* trap = dynamic_cast<const TGeoTrap*>( shape );
   const TGeoBBox* bb = dynamic_cast<const TGeoBBox*>( shape );
   if ( !bb ) {
      fwLog(fwlog::kWarning) << "CSC shape is not TGeoBBox. Ignored\n";
      shape->IsA()->Print();
      return;
   }

   // we will test 3 points on both sides ( +/- z)
   // local z is along Rho
   Double_t local[3], global[3];

   local[0]=0; local[1]=bb->GetDY(); local[2]=-bb->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=0; local[1]=-bb->GetDY(); local[2]=-bb->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=bb->GetDX(); local[1]=bb->GetDY(); local[2]=bb->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=-bb->GetDX(); local[1]=bb->GetDY(); local[2]=bb->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=bb->GetDX(); local[1]=-bb->GetDY(); local[2]=bb->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=-bb->GetDX(); local[1]=-bb->GetDY(); local[2]=bb->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );
}

void FWRPZViewGeometry::estimateProjectionSize( const Double_t* global,
                                                   double& min_rho, double& max_rho, double& min_z, double& max_z )
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

