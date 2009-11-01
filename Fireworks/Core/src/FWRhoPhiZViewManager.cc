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
// $Id: FWRhoPhiZViewManager.cc,v 1.49 2009/03/11 21:16:20 amraktad Exp $
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
#include "TEvePolygonSetProjected.h"
#include "RVersion.h"
#include "TGeoBBox.h"
#include "TGeoArb8.h"
#include "TGLEmbeddedViewer.h"
#include "TEveSelection.h"
#include "TGeoManager.h"
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"

#include "TGeoManager.h"

#include <iostream>
#include <exception>
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "vis_macros.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/FWRhoPhiZView.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBaseFactory.h"
#include "Fireworks/Core/interface/FWColorManager.h"

#include "Fireworks/Core/interface/FWEDProductRepresentationChecker.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"

#include <sstream>

//
//
// constants, enums and typedefs
//
const char* const kRhoPhiViewTypeName = "Rho Phi";
const char* const kRhoZViewTypeName = "Rho Z";

//
// static data member definitions
//

//
// constructors and destructor
//
FWRhoPhiZViewManager::FWRhoPhiZViewManager(FWGUIManager* iGUIMgr) :
   FWViewManagerBase(),
   m_rhoPhiGeomProjMgr(),
   m_rhoZGeomProjMgr(),
   m_eveStore(0),
   m_eveSelection(0),
   m_selectionManager(0),
   m_isBeingDestroyed(false)
{
   FWGUIManager::ViewBuildFunctor f;
   f = boost::bind(&FWRhoPhiZViewManager::createRhoPhiView,
                   this, _1);
   iGUIMgr->registerViewBuilder(kRhoPhiViewTypeName,f);
   f=boost::bind(&FWRhoPhiZViewManager::createRhoZView,
                 this, _1);
   iGUIMgr->registerViewBuilder(kRhoZViewTypeName,f);

   //setup geometry projections
   m_rhoPhiGeomProjMgr.reset(new TEveProjectionManager);
   //gEve->AddToListTree(m_rhoPhiGeomProjMgr,kTRUE);

   m_rhoZGeomProjMgr.reset(new TEveProjectionManager);
   m_rhoZGeomProjMgr->SetProjection(TEveProjection::kPT_RhoZ);
   //gEve->AddToListTree(m_rhoZGeomProjMgr,kTRUE);

   m_eveStore = new TEveElementList();

   //kTRUE tells it to reset the camera so we see everything
   //gEve->Redraw3D(kTRUE);
   m_eveSelection=gEve->GetSelection();
   m_eveSelection->SetPickToSelect(TEveSelection::kPS_Projectable);
   m_eveSelection->Connect("SelectionAdded(TEveElement*)","FWRhoPhiZViewManager",this,"selectionAdded(TEveElement*)");
   m_eveSelection->Connect("SelectionRemoved(TEveElement*)","FWRhoPhiZViewManager",this,"selectionRemoved(TEveElement*)");
   m_eveSelection->Connect("SelectionCleared()","FWRhoPhiZViewManager",this,"selectionCleared()");

   //create a list of the available ViewManager's
   std::set<std::string> rpzBuilders;

   std::vector<edmplugin::PluginInfo> available = FWRPZDataProxyBuilderBaseFactory::get()->available();
   std::transform(available.begin(),
                  available.end(),
                  std::inserter(rpzBuilders,rpzBuilders.begin()),
                  boost::bind(&edmplugin::PluginInfo::name_,_1));

   if(edmplugin::PluginManager::get()->categoryToInfos().end()!=edmplugin::PluginManager::get()->categoryToInfos().find(FWRPZDataProxyBuilderBaseFactory::get()->category())) {
      available = edmplugin::PluginManager::get()->categoryToInfos().find(FWRPZDataProxyBuilderBaseFactory::get()->category())->second;
      std::transform(available.begin(),
                     available.end(),
                     std::inserter(rpzBuilders,rpzBuilders.begin()),
                     boost::bind(&edmplugin::PluginInfo::name_,_1));
   }

   for(std::set<std::string>::iterator it = rpzBuilders.begin(), itEnd=rpzBuilders.end();
       it!=itEnd;
       ++it) {
      std::string::size_type first = it->find_first_of('@')+1;
      std::string purpose = it->substr(first,it->find_last_of('@')-first);
      //std::cout <<"purpose "<<purpose<<std::endl;
      m_typeToBuilder[purpose]=std::make_pair(*it,true);
   }
   
}

// FWRhoPhiZViewManager::FWRhoPhiZViewManager(const FWRhoPhiZViewManager& rhs)
// {
//    // do actual copying here;
// }

FWRhoPhiZViewManager::~FWRhoPhiZViewManager()
{
   m_isBeingDestroyed=true;
   m_rhoPhiViews.clear();
   m_rhoZViews.clear();

   m_eveStore->DestroyElements();
   m_eveStore->Destroy();
}

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
FWRhoPhiZViewManager::createRhoPhiView(TEveWindowSlot* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);

   //do geometry now so that when we open the first view we can tell it to
   // show the entire detector
   setupGeometry();

   boost::shared_ptr<FWRhoPhiZView>  pView(new FWRhoPhiZView(iParent,
                                                             kRhoPhiViewTypeName,
                                                             TEveProjection::kPT_RPhi) );
   pView->setBackgroundColor(colorManager().background());

   pView->beingDestroyed_.connect(boost::bind(&FWRhoPhiZViewManager::beingDestroyed,this,_1));
   m_rhoPhiViews.push_back(pView);
   for(TEveElement::List_i it = m_rhoPhiGeomProjMgr->BeginChildren(), itEnd = m_rhoPhiGeomProjMgr->EndChildren();
       it != itEnd;
       ++it) {
      pView->replicateGeomElement(*it);
   }
   pView->resetCamera();

   for ( std::vector<boost::shared_ptr<FWRPZDataProxyBuilderBase> >::iterator builderIter = m_builders.begin();
         builderIter != m_builders.end(); ++builderIter )  {
      (*builderIter)->attachToRhoPhiView(pView);
   }

   return pView.get();
}

FWViewBase*
FWRhoPhiZViewManager::createRhoZView(TEveWindowSlot* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);

   //do geometry now so that when we open the first view we can tell it to
   // show the entire detector
   setupGeometry();

   boost::shared_ptr<FWRhoPhiZView>  pView(new FWRhoPhiZView(iParent,
                                                             kRhoZViewTypeName,
                                                             TEveProjection::kPT_RhoZ) );
   pView->setBackgroundColor(colorManager().background());
   pView->beingDestroyed_.connect(boost::bind(&FWRhoPhiZViewManager::beingDestroyed,this,_1));
   m_rhoZViews.push_back(pView);
   for(TEveElement::List_i it = m_rhoZGeomProjMgr->BeginChildren(),
                           itEnd = m_rhoZGeomProjMgr->EndChildren();
       it != itEnd;
       ++it) {
      pView->replicateGeomElement(*it);
   }
   pView->resetCamera();
   for ( std::vector<boost::shared_ptr<FWRPZDataProxyBuilderBase> >::iterator builderIter = m_builders.begin();
         builderIter != m_builders.end(); ++builderIter )  {
      (*builderIter)->attachToRhoZView(pView);
   }
   return pView.get();
}

void
FWRhoPhiZViewManager::setupGeometry()
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   if ( m_rhoPhiGeom.empty() ) {
      makeMuonGeometryRhoPhi();
      makeTrackerGeometryRhoPhi();
   }

   if ( m_rhoZGeom.empty() ) {
      makeMuonGeometryRhoZAdvance();
      makeTrackerGeometryRhoZ();
   }
}


void
FWRhoPhiZViewManager::makeProxyBuilderFor(const FWEventItem* iItem)
{
   TypeToBuilder::iterator itFind = m_typeToBuilder.find(iItem->purpose());
   if(itFind != m_typeToBuilder.end()) {
      FWRPZDataProxyBuilderBase* builder = FWRPZDataProxyBuilderBaseFactory::get()->create(itFind->second.first);

      if(0!=builder) {
         boost::shared_ptr<FWRPZDataProxyBuilderBase> pB( builder );
         builder->setItem(iItem);
         m_builders.push_back(pB);
         pB->setViews(&m_rhoPhiViews,&m_rhoZViews);
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
}


void
FWRhoPhiZViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}
void
FWRhoPhiZViewManager::modelChangesDone()
{
   gEve->EnableRedraw();
}

void 
FWRhoPhiZViewManager::colorsChanged()
{
   for(std::vector<boost::shared_ptr<FWRhoPhiZView> >::iterator it =  m_rhoPhiViews.begin(), itEnd=m_rhoPhiViews.end();
       it != itEnd; ++it) {
      (*it)->setBackgroundColor(colorManager().background());
   }
   for(std::vector<boost::shared_ptr<FWRhoPhiZView> >::iterator it = m_rhoZViews.begin(), itEnd = m_rhoZViews.end();
       it != itEnd; ++it) {
      (*it)->setBackgroundColor(colorManager().background());
   }
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

static bool removeFrom(std::vector<boost::shared_ptr<FWRhoPhiZView> >& iViews,
                       const FWViewBase* iView) {
   typedef std::vector<boost::shared_ptr<FWRhoPhiZView> > Vect;
   for(Vect::iterator it = iViews.begin(), itEnd = iViews.end();
       it != itEnd;
       ++it) {
      if(it->get() == iView) {
         iViews.erase(it);
         return true;
      }
   }
   return false;
}

void
FWRhoPhiZViewManager::beingDestroyed(const FWViewBase* iView)
{
   //Only do this if we are NOT being called while FWRhoPhiZViewManager is being destroyed
   if(!m_isBeingDestroyed) {
      if(!removeFrom(m_rhoPhiViews,iView) ) {
         removeFrom(m_rhoZViews,iView);
      }
   }
}

//
// const member functions
//
void FWRhoPhiZViewManager::makeMuonGeometryRhoPhi()
{
   if ( !detIdToGeo() ) return;

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
         TEveGeoShape* shape = detIdToGeo()->getShape( id.rawId() );
         if ( shape ) cStation->AddElement(shape);
      }
   }

   float layer = m_rhoPhiGeomProjMgr->GetCurrentDepth();
   m_rhoPhiGeomProjMgr->SetCurrentDepth(0.);
   m_rhoPhiGeomProjMgr->ImportElements(container);
   m_rhoPhiGeomProjMgr->SetCurrentDepth(layer);

   // set background geometry visibility parameters

   TEveElementIter rhoPhiDT(m_rhoPhiGeomProjMgr.get(),"MuonRhoPhi");
   if ( rhoPhiDT.current() ) {
      m_rhoPhiGeom.push_back( rhoPhiDT.current() );
      TEveElementIter iter(rhoPhiDT.current());
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(50);
         element->SetMainColor(colorManager().geomColor(kFWMuonBarrelMainColorIndex));
         if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
            poly->SetLineColor(colorManager().geomColor(kFWMuonBarrelLineColorIndex));
         iter.next();
      }
   }

   m_eveStore->AddElement(container);
}

void FWRhoPhiZViewManager::makeMuonGeometryRhoZ()
{
   if ( !detIdToGeo() ) return;
   TEveElementList* container = new TEveElementList( "MuonRhoZ" );
   TEveElementList* dtContainer = new TEveElementList( "DT" );
   container->AddElement( dtContainer );

   for ( Int_t iWheel = -2; iWheel <= 2; ++iWheel ) {
      std::ostringstream s; s << "Wheel" << iWheel;
      TEveElementList*  cWheel  = new TEveElementList(s.str().c_str());
      dtContainer->AddElement( cWheel );
      for ( Int_t iStation=1; iStation<=4; ++iStation) {
         std::ostringstream s; s << "Station" << iStation;
         TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
         cWheel->AddElement( cStation );
         for ( Int_t iSector=1; iSector<=14; ++iSector) {
            if (iStation<4 && iSector>12) continue;
            DTChamberId id(iWheel, iStation, iSector);
            TEveGeoShape* shape = detIdToGeo()->getShape( id.rawId() );
            if ( shape ) cStation->AddElement( shape );
         }
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
                  TEveGeoShape* shape = detIdToGeo()->getShape( id.rawId() );
                  if ( shape ) cRing->AddElement( shape );
               }
               catch (... ) {}
            }
         }
      }
   }

   float layer = m_rhoZGeomProjMgr->GetCurrentDepth();
   m_rhoZGeomProjMgr->SetCurrentDepth(0.);
   m_rhoZGeomProjMgr->ImportElements( container );
   m_rhoZGeomProjMgr->SetCurrentDepth(layer);

   TEveElementIter rhoZDT(m_rhoZGeomProjMgr.get(),"DT");
   if ( rhoZDT.current() ) {
      m_rhoZGeom.push_back( rhoZDT.current() );
      TEveElementIter iter(rhoZDT.current());
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(50);
         element->SetMainColor(colorManager().geomColor(kFWMuonBarrelMainColorIndex));
         if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
            poly->SetLineColor(colorManager().geomColor(kFWMuonBarrelMainColorIndex));
         iter.next();
      }
   }

   TEveElementIter rhoZCSC(m_rhoZGeomProjMgr.get(),"CSC");
   if ( rhoZCSC.current() ) {
      m_rhoZGeom.push_back( rhoZCSC.current() );
      TEveElementIter iter(rhoZCSC.current());
      while ( iter.current() ) {
         iter.current()->SetMainTransparency(50);
         iter.current()->SetMainColor(colorManager().geomColor(kFWMuonEndCapMainColorIndex));
         if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(iter.current()) )
            poly->SetLineColor(colorManager().geomColor(kFWMuonEndCapLineColorIndex));
         iter.next();
      }
   }
   m_eveStore->AddElement(container);
}

void FWRhoPhiZViewManager::makeMuonGeometryRhoZAdvance()
{
   // lets project everything by hand
   if ( !detIdToGeo() ) return;
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
            TEveGeoShape* shape = detIdToGeo()->getShape( id.rawId() );
            if (!shape ) continue;
            estimateProjectionSizeDT( detIdToGeo()->getMatrix( id.rawId() ),
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
      for ( Int_t iStation=1; iStation<=4; ++iStation) {
         std::ostringstream s; s << "Station" << iStation;
         TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
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
                  TEveGeoShape* shape = detIdToGeo()->getShape( id.rawId() );
                  if ( !shape ) continue;
                  // get matrix from the main geometry, since detIdToGeo has reflected and
                  // rotated reference frame to get proper local coordinates
                  TGeoManager* manager = detIdToGeo()->getManager();
                  manager->cd( detIdToGeo()->getPath( id.rawId() ) );
                  const TGeoHMatrix* matrix = manager->GetCurrentMatrix();
                  estimateProjectionSizeCSC( matrix, shape->GetShape(), min_rho, max_rho, min_z, max_z );
               }
               catch (... ) {}
            }
            if ( min_rho > max_rho || min_z > max_z ) continue;
            cStation->AddElement( makeShape( s.str().c_str(), min_rho, max_rho, min_z, max_z ) );
            cStation->AddElement( makeShape( s.str().c_str(), -max_rho, -min_rho, min_z, max_z ) );
         }
      }
   }

   float layer = m_rhoZGeomProjMgr->GetCurrentDepth();
   m_rhoZGeomProjMgr->SetCurrentDepth(0.);
   m_rhoZGeomProjMgr->ImportElements( container );
   m_rhoZGeomProjMgr->SetCurrentDepth(layer);

   TEveElementIter rhoZDT(m_rhoZGeomProjMgr.get(),"DT");
   if ( rhoZDT.current() ) {
      m_rhoZGeom.push_back( rhoZDT.current() );
      TEveElementIter iter(rhoZDT.current());
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(50);
         element->SetMainColor(colorManager().geomColor(kFWMuonBarrelMainColorIndex));
         if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
            poly->SetLineColor(colorManager().geomColor(kFWMuonBarrelLineColorIndex));
         iter.next();
      }
   }

   TEveElementIter rhoZCSC(m_rhoZGeomProjMgr.get(),"CSC");
   if ( rhoZCSC.current() ) {
      m_rhoZGeom.push_back( rhoZCSC.current() );
      TEveElementIter iter(rhoZCSC.current());
      while ( iter.current() ) {
         iter.current()->SetMainTransparency(50);
         iter.current()->SetMainColor(colorManager().geomColor(kFWMuonEndCapMainColorIndex));
         if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(iter.current()) )
            poly->SetLineColor(colorManager().geomColor(kFWMuonEndCapLineColorIndex));
         iter.next();
      }
   }

   m_eveStore->AddElement(container);
}


void FWRhoPhiZViewManager::estimateProjectionSizeDT( const TGeoHMatrix* matrix, const TGeoShape* shape,
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

void FWRhoPhiZViewManager::estimateProjectionSizeCSC( const TGeoHMatrix* matrix, const TGeoShape* shape,
                                                      double& min_rho, double& max_rho, double& min_z, double& max_z )
{
   // const TGeoTrap* trap = dynamic_cast<const TGeoTrap*>( shape );
   const TGeoBBox* bb = dynamic_cast<const TGeoBBox*>( shape );
   if ( !bb ) {
      std::cout << "WARNING: CSC shape is not TGeoBBox. Ignored\n";
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

//
// static member functions
//
void FWRhoPhiZViewManager::estimateProjectionSize( const Double_t* global,
                                                   double& min_rho, double& max_rho, double& min_z, double& max_z )
{
   double rho = sqrt(global[0] *global[0]+global[1] *global[1]);
   if ( min_rho > rho ) min_rho = rho;
   if ( max_rho < rho ) max_rho = rho;
   if ( min_z > global[2] ) min_z = global[2];
   if ( max_z < global[2] ) max_z = global[2];
}


TEveGeoShape* FWRhoPhiZViewManager::makeShape( const char* name,
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


void FWRhoPhiZViewManager::makeTrackerGeometryRhoZ()
{
   TEveElementList* list = new TEveElementList( "TrackerRhoZ" );
   list->SetPickable(kFALSE);
   TEvePointSet* ref = new TEvePointSet("reference");
   ref->SetPickable(kTRUE);
   ref->SetTitle("(0,0,0)");
   ref->SetMarkerStyle(4);
   ref->SetMarkerColor(kWhite);
   list->AddElement(ref);
   ref->SetNextPoint(0.,0.,0.);
   TEveStraightLineSet* el = new TEveStraightLineSet( "outline" );
   el->SetPickable(kFALSE);
   list->AddElement(el);
   el->SetLineColor(colorManager().geomColor(kFWTrackerColorIndex));
   el->AddLine(0, 123,-300, 0, 123, 300);
   el->AddLine(0, 123, 300, 0,-123, 300);
   el->AddLine(0,-123, 300, 0,-123,-300);
   el->AddLine(0,-123,-300, 0, 123,-300);
   float layer = m_rhoZGeomProjMgr->GetCurrentDepth();
   m_rhoZGeomProjMgr->SetCurrentDepth(0.);
   m_rhoZGeomProjMgr->ImportElements( list );
   m_rhoZGeomProjMgr->ImportElements( list ); // hack
   m_rhoZGeomProjMgr->SetCurrentDepth(layer);

   m_eveStore->AddElement(list);
}

void FWRhoPhiZViewManager::makeTrackerGeometryRhoPhi()
{
   TEveStraightLineSet* el = new TEveStraightLineSet( "TrackerRhoPhi" );
   el->SetPickable(kFALSE);
   el->SetLineColor(colorManager().geomColor(kFWTrackerColorIndex));
   const unsigned int nSegments = 100;
   const double r = 123;
   for ( unsigned int i = 1; i <= nSegments; ++i )
      el->AddLine(r*sin(2*M_PI/nSegments*(i-1)), r*cos(2*M_PI/nSegments*(i-1)), 0,
                  r*sin(2*M_PI/nSegments*i), r*cos(2*M_PI/nSegments*i), 0);
   float layer = m_rhoPhiGeomProjMgr->GetCurrentDepth();
   TEvePointSet* ref = new TEvePointSet("reference");
   ref->SetPickable(kTRUE);
   ref->SetTitle("(0,0,0)");
   ref->SetMarkerStyle(4);
   ref->SetMarkerColor(kWhite);
   el->AddElement(ref);
   ref->SetNextPoint(0.,0.,0.);
   m_rhoPhiGeomProjMgr->SetCurrentDepth(0.);
   m_rhoPhiGeomProjMgr->ImportElements( el );
   m_rhoPhiGeomProjMgr->ImportElements( el ); // hack
   m_rhoPhiGeomProjMgr->SetCurrentDepth(layer);

   m_eveStore->AddElement(el);
}

FWTypeToRepresentations
FWRhoPhiZViewManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;
   const std::string kSimple("simple#");
   for(TypeToBuilder::const_iterator it = m_typeToBuilder.begin(), itEnd = m_typeToBuilder.end();
       it != itEnd;
       ++it) {
      if(it->second.first.substr(0,kSimple.size()) == kSimple) {
         returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>( new FWSimpleRepresentationChecker(
                                                                            it->second.first.substr(kSimple.size(),it->second.first.find_first_of('@')-kSimple.size()),
                                                                            it->first)));
      } else {
         returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>( new FWEDProductRepresentationChecker(
                                                                            it->second.first.substr(0,it->second.first.find_first_of('@')),
                                                                            it->first)));
      }
   }
   return returnValue;

}
