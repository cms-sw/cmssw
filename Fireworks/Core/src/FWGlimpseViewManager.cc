// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseViewManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: FWGlimpseViewManager.cc,v 1.1 2008/06/19 06:57:28 dmytro Exp $
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
#include "Fireworks/Core/interface/FWGlimpseViewManager.h"
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"

#include "TEveSelection.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilderFactory.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWGlimpseViewManager::FWGlimpseViewManager(FWGUIManager* iGUIMgr):
FWViewManagerBase(),
  m_elements(0),
  m_itemChanged(false),
  m_eveSelection(0),
  m_selectionManager(0)
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FWGlimpseViewManager::buildView,
                 this, _1);
   iGUIMgr->registerViewBuilder(FWGlimpseView::staticTypeName(), f);
   
   m_eveSelection=gEve->GetSelection();
   m_eveSelection->SetPickToSelect(TEveSelection::kPS_Projectable);
   m_eveSelection->Connect("SelectionAdded(TEveElement*)","FWGlimpseViewManager",this,"selectionAdded(TEveElement*)");
   m_eveSelection->Connect("SelectionRemoved(TEveElement*)","FWGlimpseViewManager",this,"selectionRemoved(TEveElement*)");
   m_eveSelection->Connect("SelectionCleared()","FWGlimpseViewManager",this,"selectionCleared()");

   //create a list of the available ViewManager's
   std::set<std::string> builders;
   
   std::vector<edmplugin::PluginInfo> available = FWGlimpseDataProxyBuilderFactory::get()->available();
   std::transform(available.begin(),
                  available.end(),
                  std::inserter(builders,builders.begin()),
                  boost::bind(&edmplugin::PluginInfo::name_,_1));
   
   if(edmplugin::PluginManager::get()->categoryToInfos().end()!=edmplugin::PluginManager::get()->categoryToInfos().find(FWGlimpseDataProxyBuilderFactory::get()->category())) {
      available = edmplugin::PluginManager::get()->categoryToInfos().find(FWGlimpseDataProxyBuilderFactory::get()->category())->second;
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

FWGlimpseViewManager::~FWGlimpseViewManager()
{
}

//
// member functions
//
FWViewBase* 
FWGlimpseViewManager::buildView(TGFrame* iParent)
{
   if ( ! m_elements ) m_elements = new TEveElementList("Glimpse");
   
   //? TEveManager::TRedrawDisabler disableRedraw(gEve);
   boost::shared_ptr<FWGlimpseView> view( new FWGlimpseView(iParent, m_elements) );
   view->setManager( this );
   m_views.push_back(view);
   //? pView->resetCamera();
   return view.get();
}


void 
FWGlimpseViewManager::newEventAvailable()
{
  
   if( 0==m_views.size()) return;
   
   for ( unsigned int i = 0; i < m_modelProxies.size(); ++i ) {
      if ( m_modelProxies[i].ignore ) continue;
      FWGlimpseModelProxy* proxy = & (m_modelProxies[i]);
      if ( proxy->product == 0) // first time
	{
	   TEveElementList* product(0);
	   proxy->builder->build( &product );
	   if ( ! product) {
	      printf("WARNING: proxy builder failed to initialize product for FWGlimpseViewManager. Ignored\n");
	      proxy->ignore = true;
	      continue;
	   } 
	   
	   m_elements->AddElement( product );
	   proxy->product = product;
	} else {
	   proxy->builder->build( &(proxy->product) );
	}
   }

//   std::for_each(m_views.begin(), m_views.end(),
//                 boost::bind(&FWGlimpseView::draw,_1, m_data) );
}

void 
FWGlimpseViewManager::makeProxyBuilderFor(const FWEventItem* iItem)
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
          FWGlimpseDataProxyBuilder* builder = FWGlimpseDataProxyBuilderFactory::get()->create(*builderName);
	  if(0!=builder) {
	     boost::shared_ptr<FWGlimpseDataProxyBuilder> pB( builder );
	     builder->setItem(iItem);
	     m_modelProxies.push_back(FWGlimpseModelProxy(pB) );
	  }
       }
  }
   iItem->itemChanged_.connect(boost::bind(&FWGlimpseViewManager::itemChanged,this,_1));
}

void 
FWGlimpseViewManager::newItem(const FWEventItem* iItem)
{
   makeProxyBuilderFor(iItem);
}

void 
FWGlimpseViewManager::itemChanged(const FWEventItem*) {
   m_itemChanged=true;
}
void 
FWGlimpseViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}
void 
FWGlimpseViewManager::modelChangesDone()
{
   if(m_itemChanged) {
      newEventAvailable();
   }
   m_itemChanged=false;
   gEve->EnableRedraw();
}


void
FWGlimpseViewManager::selectionAdded(TEveElement* iElement)
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
FWGlimpseViewManager::selectionRemoved(TEveElement* iElement)
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
FWGlimpseViewManager::selectionCleared()
{
   if(0!= m_selectionManager) {
      m_selectionManager->clearSelection();
   }   
}

std::vector<std::string> 
FWGlimpseViewManager::purposeForType(const std::string& iTypeName) const
{
   std::vector<std::string> returnValue;
   for(TypeToBuilders::const_iterator it = m_typeToBuilders.begin(), itEnd = m_typeToBuilders.end();
       it != itEnd;
       ++it) {
      for ( std::vector<std::string>::const_iterator builderName = it->second.begin();
	   builderName != it->second.end(); ++builderName )
      {
         if(iTypeName == builderName->substr(0,builderName->find_first_of('@'))) {
            returnValue.push_back(it->first);
         }
      }
      
   }
   return returnValue;
}

