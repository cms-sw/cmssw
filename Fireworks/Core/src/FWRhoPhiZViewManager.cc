// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRhoPhiZViewManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sat Jan  5 14:08:51 EST 2008
// $Id: FWRhoPhiZViewManager.cc,v 1.14 2008/02/03 02:49:40 dmytro Exp $
//

// system include files
#include <stdexcept>
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TEveBrowser.h"
#include "TEveGeoNode.h"
#include "TSystem.h"
#include "TEveProjectionManager.h"
#include "TEveScene.h"
#include "TGLViewer.h"
#include "TClass.h"
#include "TFile.h"
#include "TEveGeoShapeExtract.h"
#include "TEvePolygonSetProjected.h"
#include "RVersion.h"
#include "TGeoBBox.h"
#include "TGeoArb8.h"
#include "TGLEmbeddedViewer.h"

#include <iostream>
#include <exception>
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "vis_macros.h"
#include "Fireworks/Core/interface/TEveElementIter.h"

#include <sstream>

//
//
// constants, enums and typedefs
//
static
const char* const kBuilderPrefixes[] = {
   "Proxy3DBuilder",
   "ProxyRhoPhiZ2DBuilder"
};
//
// static data member definitions
//

//
// constructors and destructor
//
FWRhoPhiZViewManager::FWRhoPhiZViewManager(FWGUIManager* iGUIMgr):
  FWViewManagerBase(kBuilderPrefixes,
                    kBuilderPrefixes+sizeof(kBuilderPrefixes)/sizeof(const char*)),
  m_geom(0),
  m_rhoPhiProjMgr(0),
  m_rhoZProjMgr(0),
  m_pad(new TEvePad() ),
  m_itemChanged(false)
{
   //setup projection
   /*
   TEveViewer* nv = gEve->SpawnNewViewer("Rho Phi");
   nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TEveScene* ns = gEve->SpawnNewScene("Rho Phi");
   nv->AddScene(ns);
    */
   //Need to use an 'embedded' viewer so we can put it into the GUI
   TGLEmbeddedViewer* ev = new TGLEmbeddedViewer(iGUIMgr->parentForNextView(), m_pad);
   m_embeddedViewers.push_back(ev);
   TEveViewer* nv = new TEveViewer("Rho Phi");
   nv->SetGLViewer(ev);
   nv->IncDenyDestroy();
   iGUIMgr->addFrameHoldingAView(ev->GetFrame());
   ev->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TEveScene* ns = gEve->SpawnNewScene("Rho Phi");
   nv->AddScene(ns);
   m_viewers.push_back(nv);
   //this is needed so if a TEveElement changes this view will be informed
   gEve->AddElement(nv, gEve->GetViewers());
   
   m_rhoPhiProjMgr = new TEveProjectionManager;
   gEve->AddToListTree(m_rhoPhiProjMgr,kTRUE);
   gEve->AddElement(m_rhoPhiProjMgr,ns);
   
   /*
   nv = gEve->SpawnNewViewer("Rho Z");
   nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   ns = gEve->SpawnNewScene("Rho Z");
   nv->AddScene(ns);
   */
   ev = new TGLEmbeddedViewer(iGUIMgr->parentForNextView(), m_pad);
   m_embeddedViewers.push_back(ev);
   nv = new TEveViewer("Rho Z");
   nv->SetGLViewer(ev);
   nv->IncDenyDestroy();
   iGUIMgr->addFrameHoldingAView(ev->GetFrame());
   ev->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   ns = gEve->SpawnNewScene("Rho Z");
   nv->AddScene(ns);
   gEve->AddElement(nv, gEve->GetViewers());
   
   m_rhoZProjMgr = new TEveProjectionManager;
   m_rhoZProjMgr->SetProjection(TEveProjection::kPT_RhoZ);
   gEve->AddToListTree(m_rhoZProjMgr,kTRUE);
   gEve->AddElement(m_rhoZProjMgr,ns);
   

  //kTRUE tells it to reset the camera so we see everything 
  gEve->Redraw3D(kTRUE);  
}

// FWRhoPhiZViewManager::FWRhoPhiZViewManager(const FWRhoPhiZViewManager& rhs)
// {
//    // do actual copying here;
// }

//FWRhoPhiZViewManager::~FWRhoPhiZViewManager()
//{
//}

//
// assignment operators
//
// const FWRhoPhiZViewManager& FWRhoPhiZViewManager::operator=(const FWRhoPhiZViewManager& rhs)
// {
//   //An exception safe implementation is
//   FWRhoPhiZViewManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWRhoPhiZViewManager::rerunBuilders()
{
  using namespace std;
  if(0==gEve) {
    cout <<"Eve not initialized"<<endl;
    return;
  }

  {
     //while inside this scope, do not let
     // Eve do any redrawing
     TEveManager::TRedrawDisabler disableRedraw(gEve);

     // R-Phi projections
     
     // setup the projection
     // each projection knows what model proxies it needs
     // NOTE: this should be encapsulated and made configurable 
     //       somewhere else.
     m_rhoPhiProjMgr->DestroyElements();
     for ( std::vector<TEveElement*>::iterator element = m_rhoPhiGeom.begin(); 
	   element != m_rhoPhiGeom.end(); ++element )
       m_rhoPhiProjMgr->AddElement(*element);
     
     m_rhoZProjMgr->DestroyElements();
     for ( std::vector<TEveElement*>::iterator element = m_rhoZGeom.begin(); 
	   element != m_rhoZGeom.end(); ++element )
       m_rhoZProjMgr->AddElement(*element);
     
     // FIXME - standard way of loading geomtry failed
     // ----------- from here 
     setupGeometry();
     addElements();
  }
}

void 
FWRhoPhiZViewManager::setupGeometry()
{
   if ( ! m_geom ) {
      TFile f("tracker.root");
      if(not f.IsOpen()) {
         std::cerr <<"failed to open 'tracker.root'"<<std::endl;
         throw std::runtime_error("Failed to open 'tracker.root' geometry file");
      }
      TEveGeoShapeExtract* gse = dynamic_cast<TEveGeoShapeExtract*>(f.Get("Tracker"));
      TEveGeoShape* gsre = TEveGeoShape::ImportShapeExtract(gse,0);
      f.Close();
      m_geom = gsre;
      set_color(m_geom,kGray+3,1.,10);
      
      hide_tracker_endcap(m_geom);
      m_rhoPhiProjMgr->ImportElements(m_geom);
      
      makeMuonGeometryRhoPhi();
      // makeMuonGeometryRhoZ();
      makeMuonGeometryRhoZAdvance();
   }

}

void FWRhoPhiZViewManager::addElements()
{
   //keep track of the last element added
   TEveElement::List_i itLastRPElement = m_rhoPhiProjMgr->BeginChildren();
   TEveElement::List_i itLastRZElement = m_rhoZProjMgr->BeginChildren();
   bool rpHasMoreChildren = m_rhoPhiProjMgr->GetNChildren();
   bool rzHasMoreChildren = m_rhoZProjMgr->GetNChildren();
   int index = 0;
   while(++index < m_rhoPhiProjMgr->GetNChildren()) {++itLastRPElement;}
   index =0;
   while(++index < m_rhoZProjMgr->GetNChildren()) {++itLastRZElement;}
   
   for ( std::vector<boost::shared_ptr<FWRPZModelProxyBase> >::iterator proxy = m_modelProxies.begin();
	 proxy != m_modelProxies.end(); ++proxy )  {
      m_rhoPhiProjMgr->ImportElements((*proxy)->getRhoPhiProduct());
      m_rhoZProjMgr->ImportElements((*proxy)->getRhoZProduct());
      if(proxy == m_modelProxies.begin()) {
         if(rpHasMoreChildren) {
            ++itLastRPElement;
         }
         if(rzHasMoreChildren) {
            ++itLastRZElement;
         }
      } else {
         ++itLastRPElement;
         ++itLastRZElement;
      }
      (*proxy)->setRhoPhiProj(*itLastRPElement);
      (*proxy)->setRhoZProj(*itLastRZElement);
   }  
   
}


void 
FWRhoPhiZViewManager::newItem(const FWEventItem* iItem)
{
  TypeToBuilder::iterator itFind = m_typeToBuilder.find(iItem->name());
  if(itFind != m_typeToBuilder.end()) {
     if(itFind->second.second) {
	std::cout << "\tinterpreting as FWRPZDataProxyBuilder " << std::endl;
        FWRPZDataProxyBuilder* builder = reinterpret_cast<
        FWRPZDataProxyBuilder*>( 
                                createInstanceOf(TClass::GetClass(typeid(FWRPZDataProxyBuilder)),
                                                 itFind->second.first.c_str())
                                );
        if(0!=builder) {
           boost::shared_ptr<FWRPZDataProxyBuilder> pB( builder );
           builder->setItem(iItem);
           m_modelProxies.push_back(boost::shared_ptr<FWRPZ3DModelProxy>(new FWRPZ3DModelProxy(pB)) );
           iItem->itemChanged_.connect(boost::bind(&FWRPZModelProxyBase::itemChanged,&(*(m_modelProxies.back())),_1));
        }
     } else {
	std::cout << "\tinterpreting as FWRPZ2DDataProxyBuilder " << std::endl;
        FWRPZ2DDataProxyBuilder* builder = reinterpret_cast<
        FWRPZ2DDataProxyBuilder*>( 
                                createInstanceOf(TClass::GetClass(typeid(FWRPZ2DDataProxyBuilder)),
                                                 itFind->second.first.c_str())
                                );
        if(0!=builder) {
           boost::shared_ptr<FWRPZ2DDataProxyBuilder> pB( builder );
           builder->setItem(iItem);
           m_modelProxies.push_back(boost::shared_ptr<FWRPZ2DModelProxy>(new FWRPZ2DModelProxy(pB) ));
           iItem->itemChanged_.connect(boost::bind(&FWRPZModelProxyBase::itemChanged,&(*(m_modelProxies.back())),_1));
        }
     }
  }
   iItem->itemChanged_.connect(boost::bind(&FWRhoPhiZViewManager::itemChanged,this,_1));
}

void 
FWRhoPhiZViewManager::registerProxyBuilder(const std::string& iType,
					   const std::string& iBuilder)
{
   bool is3dType = true;
   if(iBuilder.find(kBuilderPrefixes[1]) != std::string::npos) {
      is3dType = false;
   }
   m_typeToBuilder[iType]=make_pair(iBuilder,is3dType);
}

void 
FWRhoPhiZViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}
void 
FWRhoPhiZViewManager::modelChangesDone()
{
   if(m_itemChanged) {
      rerunBuilders();
   }
   m_itemChanged=false;
   gEve->EnableRedraw();
   //gEve->Redraw3D();
}

void 
FWRhoPhiZViewManager::itemChanged(const FWEventItem*) {
   m_itemChanged=true;
}

//
// const member functions
//
void FWRhoPhiZViewManager::makeMuonGeometryRhoPhi()
{
   if ( ! detIdToGeo() ) return;
   
   TEveGeoShapeExtract* container = new TEveGeoShapeExtract( "MuonRhoPhi" );
   // rho-phi view
   Int_t iWheel = 0;
   for ( Int_t iStation=1; iStation<=4; ++iStation) {
      std::ostringstream s;
      s << "Station" << iStation;
      TEveGeoShapeExtract* cStation  = new TEveGeoShapeExtract( s.str().c_str() );
      container->AddElement( cStation );
      for ( Int_t iSector=1; iSector<=14; ++iSector) {
	 if (iStation<4 && iSector>12) continue;
	 DTChamberId id(iWheel, iStation, iSector);
	 TEveGeoShapeExtract* extract = detIdToGeo()->getExtract( id.rawId() );
	 if ( extract ) cStation->AddElement( extract );
      }
   }
   m_rhoPhiProjMgr->ImportElements( TEveGeoShape::ImportShapeExtract(container,0) );
	   
   // set background geometry visibility parameters
	
   TEveElementIter rhoPhiDT(m_rhoPhiProjMgr,"MuonRhoPhi");
   if ( rhoPhiDT.current() ) {
      m_rhoPhiGeom.push_back( rhoPhiDT.current() );
      rhoPhiDT.current()->IncDenyDestroy();
      TEveElementIter iter(rhoPhiDT.current());
      while ( TEveElement* element = iter.current() ) {
	 element->IncDenyDestroy();
	 element->SetMainTransparency(50);
	 element->SetMainColor(Color_t(TColor::GetColor("#3f0000")));
	 if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
	   poly->SetLineColor(Color_t(TColor::GetColor("#7f0000")));
	 iter.next();
      }
   }
}

void FWRhoPhiZViewManager::makeMuonGeometryRhoZ()
{
   if ( ! detIdToGeo() ) return;
   TEveGeoShapeExtract* container = new TEveGeoShapeExtract( "MuonRhoZ" );
   TEveGeoShapeExtract* dtContainer = new TEveGeoShapeExtract( "DT" );
   container->AddElement( dtContainer );
	   
   for ( Int_t iWheel = -2; iWheel <= 2; ++iWheel ) {
      std::ostringstream s; s << "Wheel" << iWheel;
      TEveGeoShapeExtract* cWheel  = new TEveGeoShapeExtract( s.str().c_str() );
      dtContainer->AddElement( cWheel );
      for ( Int_t iStation=1; iStation<=4; ++iStation) {
	 std::ostringstream s; s << "Station" << iStation;
	 TEveGeoShapeExtract* cStation  = new TEveGeoShapeExtract( s.str().c_str() );
	 cWheel->AddElement( cStation );
	 for ( Int_t iSector=1; iSector<=14; ++iSector) {
	    if (iStation<4 && iSector>12) continue;
	    DTChamberId id(iWheel, iStation, iSector);
	    TEveGeoShapeExtract* extract = detIdToGeo()->getExtract( id.rawId() );
		    if ( extract ) cStation->AddElement( extract );
	 }
      }
   }
	   
   
   TEveGeoShapeExtract* cscContainer = new TEveGeoShapeExtract( "CSC" );
   container->AddElement( cscContainer );
   for ( Int_t iEndcap = 1; iEndcap <= 2; ++iEndcap ) {// 1=forward (+Z), 2=backward(-Z)
      TEveGeoShapeExtract* cEndcap = 0;
      if (iEndcap == 1) 
	cEndcap = new TEveGeoShapeExtract( "Forward" );
      else
	cEndcap = new TEveGeoShapeExtract( "Backward" );
      cscContainer->AddElement( cEndcap );
      for ( Int_t iStation=1; iStation<=4; ++iStation) {		   
	 std::ostringstream s; s << "Station" << iStation;
	 TEveGeoShapeExtract* cStation  = new TEveGeoShapeExtract( s.str().c_str() );
	 cEndcap->AddElement( cStation );
	 for ( Int_t iRing=1; iRing<=4; ++iRing) {
	    if (iStation > 1 && iRing > 2) continue;
	    std::ostringstream s; s << "Ring" << iRing;
	    TEveGeoShapeExtract* cRing  = new TEveGeoShapeExtract( s.str().c_str() );
	    cStation->AddElement( cRing );
	    for ( Int_t iChamber=1; iChamber<=72; ++iChamber) {
	       if (iStation>1 && iChamber>36) continue;
	       Int_t iLayer = 0; // chamber 
	       // exception is thrown if parameters are not correct and I keep
	       // forgetting how many chambers we have in each ring.
	       try {
		  CSCDetId id(iEndcap, iStation, iRing, iChamber, iLayer);
		  TEveGeoShapeExtract* extract = detIdToGeo()->getExtract( id.rawId() );
		  if ( extract )  cRing->AddElement( extract );
	       }
	       catch ( ... ) {} 
	    }
	 }
      }
   }
   m_rhoZProjMgr->ImportElements( TEveGeoShape::ImportShapeExtract(container,0) );
   
   TEveElementIter rhoZDT(m_rhoZProjMgr,"DT");
   if ( rhoZDT.current() ) {
      m_rhoZGeom.push_back( rhoZDT.current() );
      rhoZDT.current()->IncDenyDestroy();
      TEveElementIter iter(rhoZDT.current());
      while ( TEveElement* element = iter.current() ) {
	 element->IncDenyDestroy();
	 element->SetMainTransparency(50);
	 element->SetMainColor(Color_t(TColor::GetColor("#3f0000")));
	 if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
	   poly->SetLineColor(Color_t(TColor::GetColor("#3f0000")));
	 iter.next();
      }
   }
	
   TEveElementIter rhoZCSC(m_rhoZProjMgr,"CSC");
   if ( rhoZCSC.current() ) {
      m_rhoZGeom.push_back( rhoZCSC.current() );
      rhoZCSC.current()->IncDenyDestroy();
      TEveElementIter iter(rhoZCSC.current());
      while ( iter.current() ) {
	 iter.current()->SetMainTransparency(50);
	 iter.current()->SetMainColor(Color_t(TColor::GetColor("#00003f")));
	 if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(iter.current()) )
	   poly->SetLineColor(Color_t(TColor::GetColor("#00003f")));
	 iter.next();
      }
   }
   
}

void FWRhoPhiZViewManager::makeMuonGeometryRhoZAdvance()
{
   // lets project everything by hand
   if ( ! detIdToGeo() ) return;
   TEveGeoShapeExtract* container = new TEveGeoShapeExtract( "MuonRhoZ" );
   TEveGeoShapeExtract* dtContainer = new TEveGeoShapeExtract( "DT" );
   container->AddElement( dtContainer );
	   
   for ( Int_t iWheel = -2; iWheel <= 2; ++iWheel ) {
      std::ostringstream s; s << "Wheel" << iWheel;
      TEveGeoShapeExtract* cWheel  = new TEveGeoShapeExtract( s.str().c_str() );
      dtContainer->AddElement( cWheel );
      for ( Int_t iStation=1; iStation<=4; ++iStation) {
	 std::ostringstream s; s << "Station" << iStation;
	 double min_rho(1000), max_rho(0), min_z(2000), max_z(-2000);
	 
	 for ( Int_t iSector=1; iSector<=14; ++iSector) {
	    if (iStation<4 && iSector>12) continue;
	    DTChamberId id(iWheel, iStation, iSector);
	    TEveGeoShapeExtract* extract = detIdToGeo()->getExtract( id.rawId() );
	    if (! extract ) continue;
	    estimateProjectionSizeDT( detIdToGeo()->getMatrix( id.rawId() ), 
				      extract->GetShape(), min_rho, max_rho, min_z, max_z );
	 }
	 if ( min_rho > max_rho || min_z > max_z ) continue;
	 cWheel->AddElement( makeShapeExtract( s.str().c_str(), min_rho, max_rho, min_z, max_z ) );
	 cWheel->AddElement( makeShapeExtract( s.str().c_str(), -max_rho, -min_rho, min_z, max_z ) );
      }
   }

   TEveGeoShapeExtract* cscContainer = new TEveGeoShapeExtract( "CSC" );
   container->AddElement( cscContainer );
   for ( Int_t iEndcap = 1; iEndcap <= 2; ++iEndcap ) {// 1=forward (+Z), 2=backward(-Z)
      TEveGeoShapeExtract* cEndcap = 0;
      if (iEndcap == 1) 
	cEndcap = new TEveGeoShapeExtract( "Forward" );
      else
	cEndcap = new TEveGeoShapeExtract( "Backward" );
      cscContainer->AddElement( cEndcap );
      for ( Int_t iStation=1; iStation<=4; ++iStation) {		   
	 std::ostringstream s; s << "Station" << iStation;
	 TEveGeoShapeExtract* cStation  = new TEveGeoShapeExtract( s.str().c_str() );
	 cEndcap->AddElement( cStation );
	 for ( Int_t iRing=1; iRing<=4; ++iRing) {
	    if (iStation > 1 && iRing > 2) continue;
	    std::ostringstream s; s << "Ring" << iRing;
	    double min_rho(1000), max_rho(0), min_z(2000), max_z(-2000);
	    for ( Int_t iChamber=1; iChamber<=72; ++iChamber) {
	       if (iStation>1 && iChamber>36) continue;
	       Int_t iLayer = 0; // chamber 
	       // exception is thrown if parameters are not correct and I keep
	       // forgetting how many chambers we have in each ring.
	       try {
		  CSCDetId id(iEndcap, iStation, iRing, iChamber, iLayer);
		  TEveGeoShapeExtract* extract = detIdToGeo()->getExtract( id.rawId() );
		  if ( !extract ) continue;
		  gGeoManager->cd( detIdToGeo()->getPath( id.rawId() ) );
		  TGeoHMatrix* matrix = gGeoManager->GetCurrentMatrix();
		  estimateProjectionSizeCSC( matrix, extract->GetShape(), min_rho, max_rho, min_z, max_z );
	       }
	       catch ( ... ) {} 
	    }
	    if ( min_rho > max_rho || min_z > max_z ) continue;
	    cStation->AddElement( makeShapeExtract( s.str().c_str(), min_rho, max_rho, min_z, max_z ) );
	    cStation->AddElement( makeShapeExtract( s.str().c_str(), -max_rho, -min_rho, min_z, max_z ) );
	 }
      }
   }
   m_rhoZProjMgr->ImportElements( TEveGeoShape::ImportShapeExtract(container,0) );
   
   TEveElementIter rhoZDT(m_rhoZProjMgr,"DT");
   if ( rhoZDT.current() ) {
      m_rhoZGeom.push_back( rhoZDT.current() );
      rhoZDT.current()->IncDenyDestroy();
      TEveElementIter iter(rhoZDT.current());
      while ( TEveElement* element = iter.current() ) {
	 element->IncDenyDestroy();
	 element->SetMainTransparency(50);
	 element->SetMainColor(Color_t(TColor::GetColor("#3f0000")));
	 if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
	   poly->SetLineColor(Color_t(TColor::GetColor("#7f0000")));
	 iter.next();
      }
   }
   
   TEveElementIter rhoZCSC(m_rhoZProjMgr,"CSC");
   if ( rhoZCSC.current() ) {
      m_rhoZGeom.push_back( rhoZCSC.current() );
      rhoZCSC.current()->IncDenyDestroy();
      TEveElementIter iter(rhoZCSC.current());
      while ( iter.current() ) {
	 iter.current()->SetMainTransparency(50);
	 iter.current()->SetMainColor(Color_t(TColor::GetColor("#00003f")));
	 if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(iter.current()) )
	   poly->SetLineColor(Color_t(TColor::GetColor("#00007f")));
	 iter.next();
      }
   }

}


void FWRhoPhiZViewManager::estimateProjectionSizeDT( const TGeoHMatrix* matrix, const TGeoShape* shape,
						   double& min_rho, double& max_rho, double& min_z, double& max_z )
{
   const TGeoBBox* box = dynamic_cast<const TGeoBBox*>( shape );
   if ( ! box ) return;

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

void FWRhoPhiZViewManager::estimateProjectionSizeCSC( const TGeoHMatrix* matrix, const TGeoShape* shape,
						      double& min_rho, double& max_rho, double& min_z, double& max_z )
{
   const TGeoTrap* trap = dynamic_cast<const TGeoTrap*>( shape );
   if ( ! trap ) return;

   // we will test 3 points on both sides ( +/- z)
   // local z is along Rho
   Double_t local[3], global[3];
   
   local[0]=0; local[1]=trap->GetH1(); local[2]=-trap->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=0; local[1]=-trap->GetH1(); local[2]=-trap->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=trap->GetTl2(); local[1]=trap->GetH2(); local[2]=trap->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=-trap->GetTl2(); local[1]=trap->GetH2(); local[2]=trap->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=trap->GetTl2(); local[1]=-trap->GetH2(); local[2]=trap->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );

   local[0]=-trap->GetTl2(); local[1]=-trap->GetH2(); local[2]=trap->GetDZ();
   matrix->LocalToMaster(local,global);
   estimateProjectionSize( global, min_rho, max_rho, min_z, max_z );
}

//
// static member functions
//

void
FWRPZModelProxyBase::itemChanged(const FWEventItem* iItem)
{
   this->itemChangedImp(iItem);
}

void
FWRPZ3DModelProxy::itemChangedImp(const FWEventItem*)
{
   m_mustRebuild=true;
}

TEveElementList* 
FWRPZ3DModelProxy::getProduct() const
{
   if(m_mustRebuild) {
      m_builder->build(&m_product);
      m_mustRebuild=false;
   }
   return m_product;
}

TEveElementList* 
FWRPZ3DModelProxy::getRhoPhiProduct() const
{
   return getProduct();
}

TEveElementList* 
FWRPZ3DModelProxy::getRhoZProduct() const
{
   return getProduct();
}

void
FWRPZ3DModelProxy::setRhoPhiProj(TEveElement* iElement)
{
   m_builder->setRhoPhiProj(iElement);
}

void
FWRPZ3DModelProxy::setRhoZProj(TEveElement* iElement)
{
   m_builder->setRhoZProj(iElement);
}
void
FWRPZ2DModelProxy::itemChangedImp(const FWEventItem*)
{
   m_mustRebuildRhoPhi=true;
   m_mustRebuildRhoZ=true;
}
TEveElementList* 
FWRPZ2DModelProxy::getRhoPhiProduct() const
{
   if(m_mustRebuildRhoPhi) {
      m_builder->buildRhoPhi(&m_rhoPhiProduct);
      m_mustRebuildRhoPhi=false;
   }
   return m_rhoPhiProduct;
}

TEveElementList* 
FWRPZ2DModelProxy::getRhoZProduct() const
{
   if(m_mustRebuildRhoZ) {
      m_builder->buildRhoZ(&m_rhoZProduct);
      m_mustRebuildRhoZ=false;
   }
   return m_rhoZProduct;
}

void
FWRPZ2DModelProxy::setRhoPhiProj(TEveElement* iElement)
{
   m_builder->setRhoPhiProj(iElement);
}

void
FWRPZ2DModelProxy::setRhoZProj(TEveElement* iElement)
{
   m_builder->setRhoZProj(iElement);
}
void FWRhoPhiZViewManager::estimateProjectionSize( const Double_t* global,
						   double& min_rho, double& max_rho, double& min_z, double& max_z )
{
   double rho = sqrt(global[0]*global[0]+global[1]*global[1]);
   if ( min_rho > rho ) min_rho = rho;
   if ( max_rho < rho ) max_rho = rho;
   if ( min_z > global[2] ) min_z = global[2];
   if ( max_z < global[2] ) max_z = global[2];
}
		 
		 
TEveGeoShapeExtract* FWRhoPhiZViewManager::makeShapeExtract( const char* name, 
							     double min_rho, double max_rho, double min_z, double max_z )
{
   TEveTrans t;
   t(1,1) = 1; t(1,2) = 0; t(1,3) = 0;
   t(2,1) = 0; t(2,2) = 1; t(2,3) = 0;
   t(3,1) = 0; t(3,2) = 0; t(3,3) = 1;
   t(1,4) = 0; t(2,4) = (min_rho+max_rho)/2; t(3,4) = (min_z+max_z)/2;
   
   TEveGeoShapeExtract* extract = new TEveGeoShapeExtract(name);
   extract->SetTrans(t.Array());
   
   extract->SetRnrSelf(kTRUE);
   extract->SetRnrElements(kTRUE);
   TGeoBBox* box = new TGeoBBox( 0, (max_rho-min_rho)/2, (max_z-min_z)/2 ); 
   extract->SetShape( box );
   return extract;
}
