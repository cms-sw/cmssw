// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DViewManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: FW3DViewManager.cc,v 1.12 2008/11/14 16:36:51 chrjones Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include <algorithm>
#include "TView.h"
#include "TList.h"
#include "TEveManager.h"
#include "TClass.h"
#include "TColor.h"
#include "TRootEmbeddedCanvas.h"
#include "TEveCaloData.h"
#include "TEveElement.h"
#include "TROOT.h"

// user include files
#include "Fireworks/Core/interface/FW3DViewManager.h"
#include "Fireworks/Core/interface/FW3DView.h"
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"

#include "TEveSelection.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

#include "Fireworks/Core/interface/FW3DDataProxyBuilderFactory.h"

#include "Fireworks/Core/interface/FWEDProductRepresentationChecker.h"
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "TEveGeoNode.h"
#include "TEveScene.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DViewManager::FW3DViewManager(FWGUIManager* iGUIMgr):
FWViewManagerBase(),
  m_elements("3D"),
  m_eveSelection(0),
  m_selectionManager(0)
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FW3DViewManager::buildView,
                 this, _1);
   iGUIMgr->registerViewBuilder(FW3DView::staticTypeName(), f);

   /*
   m_eveSelection=gEve->GetSelection();
   m_eveSelection->SetPickToSelect(TEveSelection::kPS_Projectable);
   m_eveSelection->Connect("SelectionAdded(TEveElement*)","FW3DViewManager",this,"selectionAdded(TEveElement*)");
   m_eveSelection->Connect("SelectionRemoved(TEveElement*)","FW3DViewManager",this,"selectionRemoved(TEveElement*)");
   m_eveSelection->Connect("SelectionCleared()","FW3DViewManager",this,"selectionCleared()");
    */

   //create a list of the available ViewManager's
   std::set<std::string> builders;

   std::vector<edmplugin::PluginInfo> available = FW3DDataProxyBuilderFactory::get()->available();
   std::transform(available.begin(),
                  available.end(),
                  std::inserter(builders,builders.begin()),
                  boost::bind(&edmplugin::PluginInfo::name_,_1));

   if(edmplugin::PluginManager::get()->categoryToInfos().end()!=edmplugin::PluginManager::get()->categoryToInfos().find(FW3DDataProxyBuilderFactory::get()->category())) {
      available = edmplugin::PluginManager::get()->categoryToInfos().find(FW3DDataProxyBuilderFactory::get()->category())->second;
      std::transform(available.begin(),
                     available.end(),
                     std::inserter(builders,builders.begin()),
                     boost::bind(&edmplugin::PluginInfo::name_,_1));
   }

   for(std::set<std::string>::iterator it = builders.begin(), itEnd=builders.end();
       it!=itEnd;
       ++it) {
      std::string::size_type first = it->find_first_of('@')+1;
      std::string  purpose = it->substr(first,it->find_last_of('@')-first);
      m_typeToBuilders[purpose].push_back(*it);
   }

}

FW3DViewManager::~FW3DViewManager()
{
}

//
// member functions
//
FWViewBase*
FW3DViewManager::buildView(TGFrame* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);
   boost::shared_ptr<FW3DView> view( new FW3DView(iParent, &m_elements) );
   m_views.push_back(view);
   //? pView->resetCamera();
   if(1 == m_views.size()) {
      for(std::vector<boost::shared_ptr<FW3DDataProxyBuilder> >::iterator it
          =m_builders.begin(), itEnd = m_builders.end();
          it != itEnd;
          ++it) {
         (*it)->setHaveAWindow(true);
      }
   }
   makeGeometry(view->getScene());
   view->beingDestroyed_.connect(boost::bind(&FW3DViewManager::beingDestroyed,this,_1));
   return view.get();
}
void
FW3DViewManager::beingDestroyed(const FWViewBase* iView)
{

   if(1 == m_views.size()) {
      for(std::vector<boost::shared_ptr<FW3DDataProxyBuilder> >::iterator it
          =m_builders.begin(), itEnd = m_builders.end();
          it != itEnd;
          ++it) {
         (*it)->setHaveAWindow(false);
      }
   }
   for(std::vector<boost::shared_ptr<FW3DView> >::iterator it=
       m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
      if(it->get() == iView) {
         m_views.erase(it);
         return;
      }
   }
}

void
FW3DViewManager::makeProxyBuilderFor(const FWEventItem* iItem)
{
   if(0==m_selectionManager) {
      //std::cout <<"got selection manager"<<std::endl;
      m_selectionManager = iItem->selectionManager();
   }
  TypeToBuilders::iterator itFind = m_typeToBuilders.find(iItem->purpose());
  if(itFind != m_typeToBuilders.end()) {
     for ( std::vector<std::string>::const_iterator builderName = itFind->second.begin();
	   builderName != itFind->second.end(); ++builderName )
       {
          FW3DDataProxyBuilder* builder = FW3DDataProxyBuilderFactory::get()->create(*builderName);
	  if(0!=builder) {
	     boost::shared_ptr<FW3DDataProxyBuilder> pB( builder );
	     builder->setItem(iItem);
             builder->setHaveAWindow(!m_views.empty());
             m_elements.AddElement(builder->usedInScene());
             m_builders.push_back(pB);
	  }
       }
  }
}

void
FW3DViewManager::newItem(const FWEventItem* iItem)
{
   makeProxyBuilderFor(iItem);
}

void
FW3DViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}
void
FW3DViewManager::modelChangesDone()
{
   gEve->EnableRedraw();
}


void
FW3DViewManager::selectionAdded(TEveElement* iElement)
{
   //std::cout <<"selection added"<<std::endl;
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
FW3DViewManager::selectionRemoved(TEveElement* iElement)
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
FW3DViewManager::selectionCleared()
{
   if(0!= m_selectionManager) {
      m_selectionManager->clearSelection();
   }
}

FWTypeToRepresentations
FW3DViewManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;

   for(TypeToBuilders::const_iterator it = m_typeToBuilders.begin(), itEnd = m_typeToBuilders.end();
       it != itEnd;
       ++it) {
      for ( std::vector<std::string>::const_iterator builderName = it->second.begin();
	   builderName != it->second.end(); ++builderName )
      {
         returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>( new FWEDProductRepresentationChecker(
                                                                                                              builderName->substr(0,builderName->find_first_of('@')),
                                                                                                              it->first)));
      }

   }
   return returnValue;
}

void FW3DViewManager::makeGeometry( TEveElement* parent )
{
   if ( ! detIdToGeo() ) {
      std::cout << "Warning: cannot get geometry to rendered detector outline. Skipped" << std::endl;
      return;
   }
   
   // rho-phi view
   TEveElementList* container = new TEveElementList( "DT" );
   gEve->AddElement( container, parent );
   for ( Int_t iWheel = -2; iWheel <= 2; ++iWheel)
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
	       shape->SetMainTransparency(90);
	       if ( shape ) cStation->AddElement(shape);
	    }
       }
/*
   // set background geometry visibility parameters

   TEveElementIter rhoPhiDT(m_rhoPhiGeomProjMgr.get(),"MuonRhoPhi");
   if ( rhoPhiDT.current() ) {
      m_rhoPhiGeom.push_back( rhoPhiDT.current() );
      TEveElementIter iter(rhoPhiDT.current());
      while ( TEveElement* element = iter.current() ) {
	 element->SetMainTransparency(50);
	 element->SetMainColor(Color_t(TColor::GetColor("#3f0000")));
	 if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
	   poly->SetLineColor(Color_t(TColor::GetColor("#7f0000")));
	 iter.next();
      }
   }
*/
}

