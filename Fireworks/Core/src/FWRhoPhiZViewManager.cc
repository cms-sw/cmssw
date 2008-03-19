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
// $Id: FWRhoPhiZViewManager.cc,v 1.23 2008/03/14 21:11:01 chrjones Exp $
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
#include "TEveSelection.h"

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
#include "Fireworks/Core/interface/FWRhoPhiZView.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

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
static
const char* const kRhoPhiViewTypeName = "Rho Phi";
const char* const kRhoZViewTypeName = "Rho Z";
//
// static data member definitions
//

//
// constructors and destructor
//
FWRhoPhiZViewManager::FWRhoPhiZViewManager(FWGUIManager* iGUIMgr):
  FWViewManagerBase(kBuilderPrefixes,
                    kBuilderPrefixes+sizeof(kBuilderPrefixes)/sizeof(const char*)),
  m_rhoPhiGeomProjMgr(0),
  m_rhoZGeomProjMgr(0),
  //m_pad(new TEvePad() ),
  m_itemChanged(false),
  m_eveSelection(0),
  m_selectionManager(0)
{
   FWGUIManager::ViewBuildFunctor f;
   f = boost::bind(&FWRhoPhiZViewManager::createRhoPhiView,
                   this, _1);
   iGUIMgr->registerViewBuilder(kRhoPhiViewTypeName,f);
   f=boost::bind(&FWRhoPhiZViewManager::createRhoZView,
                 this, _1);
   iGUIMgr->registerViewBuilder(kRhoZViewTypeName,f);

   //setup geometry projections
   m_rhoPhiGeomProjMgr = new TEveProjectionManager;
   //gEve->AddToListTree(m_rhoPhiGeomProjMgr,kTRUE);
   
   m_rhoZGeomProjMgr = new TEveProjectionManager;
   m_rhoZGeomProjMgr->SetProjection(TEveProjection::kPT_RhoZ);
   //gEve->AddToListTree(m_rhoZGeomProjMgr,kTRUE);
   
   //kTRUE tells it to reset the camera so we see everything 
   //gEve->Redraw3D(kTRUE);  
   m_eveSelection=gEve->GetSelection();
   m_eveSelection->SetPickToSelect(TEveSelection::kPS_Projectable);
   m_eveSelection->Connect("SelectionAdded(TEveElement*)","FWRhoPhiZViewManager",this,"selectionAdded(TEveElement*)");
   m_eveSelection->Connect("SelectionRemoved(TEveElement*)","FWRhoPhiZViewManager",this,"selectionRemoved(TEveElement*)");
   m_eveSelection->Connect("SelectionCleared()","FWRhoPhiZViewManager",this,"selectionCleared()");
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
FWViewBase* 
FWRhoPhiZViewManager::createRhoPhiView(TGFrame* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);
   //do geometry now so that when we open the first view we can tell it to 
   // show the entire detector
   setupGeometry();
   
   boost::shared_ptr<FWRhoPhiZView>  pView(new FWRhoPhiZView(iParent,
                                                             kRhoPhiViewTypeName,
                                                             TEveProjection::kPT_RPhi) );
   m_rhoPhiViews.push_back(pView);
   for(TEveElement::List_i it = m_rhoPhiGeomProjMgr->BeginChildren(), itEnd = m_rhoPhiGeomProjMgr->EndChildren();
       it != itEnd;
       ++it) {
      pView->replicateGeomElement(*it);
   }
   pView->resetCamera();
   return pView.get();
}

FWViewBase* 
FWRhoPhiZViewManager::createRhoZView(TGFrame* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);
  //do geometry now so that when we open the first view we can tell it to 
   // show the entire detector
   setupGeometry();
   
   boost::shared_ptr<FWRhoPhiZView>  pView(new FWRhoPhiZView(iParent,
                                                             kRhoZViewTypeName,
                                                             TEveProjection::kPT_RhoZ) );
   m_rhoZViews.push_back(pView);
   for(TEveElement::List_i it = m_rhoZGeomProjMgr->BeginChildren(), 
       itEnd = m_rhoZGeomProjMgr->EndChildren();
       it != itEnd;
       ++it) {
      pView->replicateGeomElement(*it);
   }
   pView->resetCamera();
   return pView.get();
}


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

     std::for_each(m_rhoPhiViews.begin(),
                   m_rhoPhiViews.end(),
                   boost::bind(&FWRhoPhiZView::destroyElements, _1));
     std::for_each(m_rhoZViews.begin(),
                   m_rhoZViews.end(),
                   boost::bind(&FWRhoPhiZView::destroyElements, _1));
     
     addElements();
  }
}

void 
FWRhoPhiZViewManager::setupGeometry()
{
   if ( m_rhoPhiGeom.empty() ) makeMuonGeometryRhoPhi();
   // makeMuonGeometryRhoZ();
   if ( m_rhoZGeom.empty() ) makeMuonGeometryRhoZAdvance();
}

void FWRhoPhiZViewManager::addElements()
{
   for ( std::vector<boost::shared_ptr<FWRPZModelProxyBase> >::iterator proxy = m_modelProxies.begin();
	 proxy != m_modelProxies.end(); ++proxy )  {
      (*proxy)->clearRhoPhiProjs();
      for(std::vector<boost::shared_ptr<FWRhoPhiZView> >::iterator it = m_rhoPhiViews.begin(),
          itEnd = m_rhoPhiViews.end();
          it != itEnd;
          ++it) {
         (*proxy)->addRhoPhiProj( (*it)->importElements((*proxy)->getRhoPhiProduct(),(*proxy)->layer()));
      }
      (*proxy)->clearRhoZProjs();
      for(std::vector<boost::shared_ptr<FWRhoPhiZView> >::iterator it = m_rhoZViews.begin(),
          itEnd = m_rhoZViews.end();
          it != itEnd;
          ++it) {
         (*proxy)->addRhoZProj( (*it)->importElements((*proxy)->getRhoZProduct(),(*proxy)->layer()));
      }
   }  
   
}

void
FWRhoPhiZViewManager::makeProxyBuilderFor(const FWEventItem* iItem)
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
}

void 
FWRhoPhiZViewManager::newItem(const FWEventItem* iItem)
{
  if(0==m_selectionManager) {
     //std::cout <<"got selection manager"<<std::endl;
     m_selectionManager = iItem->selectionManager();
  }
   makeProxyBuilderFor(iItem);
   iItem->itemChanged_.connect(boost::bind(&FWRhoPhiZViewManager::itemChanged,this,_1));
}

void 
FWRhoPhiZViewManager::registerProxyBuilder(const std::string& iType,
					   const std::string& iBuilder,
                                           const FWEventItem* iItem)
{
   bool is3dType = true;
   if(iBuilder.find(kBuilderPrefixes[1]) != std::string::npos) {
      is3dType = false;
   }
   m_typeToBuilder[iType]=make_pair(iBuilder,is3dType);
   
   //has the item already been registered? If so then we need to make the proxy builder
   if(iItem!=0) {
      std::cout <<"item "<<iType<<" registered before proxy builder"<<std::endl;
       makeProxyBuilderFor(iItem);
   }
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

void
FWRhoPhiZViewManager::selectionAdded(TEveElement* iElement)
{
   //std::cout <<"selection added "<<iElement<< std::endl;
   if(0!=iElement) {
      void* userData=iElement->GetUserData();
      //std::cout <<"  user data "<<userData<<std::endl;
      if(0 != userData) {
         FWModelId* id = static_cast<FWModelId*>(userData);
         if( not id->item()->modelInfo(id->index()).isSelected() ) {
            bool last = m_eveSelection->BlockSignals(kTRUE);
            //std::cout <<"   selecting"<<std::endl;

            id->select();
            m_eveSelection->BlockSignals(last);
         }
      }
   }
}

void
FWRhoPhiZViewManager::selectionRemoved(TEveElement* iElement)
{
   //std::cout <<"selection removed"<<std::endl;
   if(0!=iElement) {
      void* userData=iElement->GetUserData();
      if(0 != userData) {
         FWModelId* id = static_cast<FWModelId*>(userData);
         if( id->item()->modelInfo(id->index()).isSelected() ) {
            bool last = m_eveSelection->BlockSignals(kTRUE);
            //std::cout <<"   removing"<<std::endl;
            id->unselect();
            m_eveSelection->BlockSignals(last);
         }
      }
   }
   
}

void
FWRhoPhiZViewManager::selectionCleared()
{
   if(0!= m_selectionManager) {
      m_selectionManager->clearSelection();
   }   
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
   TEveElement* el = TEveGeoShape::ImportShapeExtract(container,0);
   el->IncDenyDestroy();
   float layer = m_rhoPhiGeomProjMgr->GetCurrentDepth();
   m_rhoPhiGeomProjMgr->SetCurrentDepth(0.);
   m_rhoPhiGeomProjMgr->ImportElements( el );
   m_rhoPhiGeomProjMgr->SetCurrentDepth(layer);
	   
   // set background geometry visibility parameters
	
   TEveElementIter rhoPhiDT(m_rhoPhiGeomProjMgr,"MuonRhoPhi");
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
   TEveElement* el = TEveGeoShape::ImportShapeExtract(container,0);
   el->IncDenyDestroy();
   float layer = m_rhoZGeomProjMgr->GetCurrentDepth();
   m_rhoZGeomProjMgr->SetCurrentDepth(0.);
   m_rhoZGeomProjMgr->ImportElements( el );
   m_rhoZGeomProjMgr->SetCurrentDepth(layer);
   
   TEveElementIter rhoZDT(m_rhoZGeomProjMgr,"DT");
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
	
   TEveElementIter rhoZCSC(m_rhoZGeomProjMgr,"CSC");
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
   TEveElement* el = TEveGeoShape::ImportShapeExtract(container,0);
   el->IncDenyDestroy();
   float layer = m_rhoZGeomProjMgr->GetCurrentDepth();
   m_rhoZGeomProjMgr->SetCurrentDepth(0.);
   m_rhoZGeomProjMgr->ImportElements( el );
   m_rhoZGeomProjMgr->SetCurrentDepth(layer);
   
   TEveElementIter rhoZDT(m_rhoZGeomProjMgr,"DT");
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
   
   TEveElementIter rhoZCSC(m_rhoZGeomProjMgr,"CSC");
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
   if(0!=iItem) {
      m_layer = iItem->layer();
   }
   this->itemChangedImp(iItem);
}

float
FWRPZModelProxyBase::layer() const
{
   return m_layer;
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
FWRPZ3DModelProxy::addRhoPhiProj(TEveElement* iElement)
{
   m_builder->addRhoPhiProj(iElement);
}

void
FWRPZ3DModelProxy::addRhoZProj(TEveElement* iElement)
{
   m_builder->addRhoZProj(iElement);
}

void
FWRPZ3DModelProxy::clearRhoPhiProjs()
{
   m_builder->clearRhoPhiProjs();
}

void
FWRPZ3DModelProxy::clearRhoZProjs()
{
   m_builder->clearRhoZProjs();
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
FWRPZ2DModelProxy::addRhoPhiProj(TEveElement* iElement)
{
   m_builder->addRhoPhiProj(iElement);
}

void
FWRPZ2DModelProxy::addRhoZProj(TEveElement* iElement)
{
   m_builder->addRhoZProj(iElement);
}

void
FWRPZ2DModelProxy::clearRhoPhiProjs()
{
   m_builder->clearRhoPhiProjs();
}

void
FWRPZ2DModelProxy::clearRhoZProjs()
{
   m_builder->clearRhoZProjs();
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
