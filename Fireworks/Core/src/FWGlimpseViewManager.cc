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
// $Id: FWGlimpseViewManager.cc,v 1.16 2009/03/11 21:16:20 amraktad Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include <algorithm>
#include "TView.h"
#include "TList.h"
#include "TEveManager.h"
#include "TClass.h"

// user include files
#include "Fireworks/Core/interface/FWGlimpseViewManager.h"
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"

#include "TEveSelection.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilderFactory.h"

#include "Fireworks/Core/interface/FWEDProductRepresentationChecker.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWGlimpseViewManager::FWGlimpseViewManager(FWGUIManager* iGUIMgr) :
   FWViewManagerBase(),
   m_elements("Glimpse"),
   m_eveSelection(0),
   m_selectionManager(0),
   m_scaler(1.0)
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FWGlimpseViewManager::buildView,
                 this, _1);
   iGUIMgr->registerViewBuilder(FWGlimpseView::staticTypeName(), f);

   /*
      m_eveSelection=gEve->GetSelection();
      m_eveSelection->SetPickToSelect(TEveSelection::kPS_Projectable);
      m_eveSelection->Connect("SelectionAdded(TEveElement*)","FWGlimpseViewManager",this,"selectionAdded(TEveElement*)");
      m_eveSelection->Connect("SelectionRemoved(TEveElement*)","FWGlimpseViewManager",this,"selectionRemoved(TEveElement*)");
      m_eveSelection->Connect("SelectionCleared()","FWGlimpseViewManager",this,"selectionCleared()");
    */

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
      std::string purpose = it->substr(first,it->find_last_of('@')-first);
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
FWGlimpseViewManager::buildView(TEveWindowSlot* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);
   boost::shared_ptr<FWGlimpseView> view( new FWGlimpseView(iParent, &m_elements,&m_scaler) );
   view->setBackgroundColor(colorManager().background());
   m_views.push_back(view);
   //? pView->resetCamera();
   if(1 == m_views.size()) {
      for(std::vector<boost::shared_ptr<FWGlimpseDataProxyBuilder> >::iterator it
             =m_builders.begin(), itEnd = m_builders.end();
          it != itEnd;
          ++it) {
         (*it)->setHaveAWindow(true);
      }
   }
   view->beingDestroyed_.connect(boost::bind(&FWGlimpseViewManager::beingDestroyed,this,_1));
   return view.get();
}
void
FWGlimpseViewManager::beingDestroyed(const FWViewBase* iView)
{

   if(1 == m_views.size()) {
      for(std::vector<boost::shared_ptr<FWGlimpseDataProxyBuilder> >::iterator it
             =m_builders.begin(), itEnd = m_builders.end();
          it != itEnd;
          ++it) {
         (*it)->setHaveAWindow(false);
      }
   }
   for(std::vector<boost::shared_ptr<FWGlimpseView> >::iterator it=
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
            builder->setHaveAWindow(!m_views.empty());
            builder->setScaler(&m_scaler);
            m_elements.AddElement(builder->usedInScene());
            m_builders.push_back(pB);
         }
      }
   }
}

void
FWGlimpseViewManager::newItem(const FWEventItem* iItem)
{
   makeProxyBuilderFor(iItem);
}

void
FWGlimpseViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}
void
FWGlimpseViewManager::modelChangesDone()
{
   gEve->EnableRedraw();
}
void
FWGlimpseViewManager::colorsChanged()
{
   for(std::vector<boost::shared_ptr<FWGlimpseView> >::iterator it=
       m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
      (*it)->setBackgroundColor(colorManager().background());
   }
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

FWTypeToRepresentations
FWGlimpseViewManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;
   const std::string kSimple("simple#");

   for(TypeToBuilders::const_iterator it = m_typeToBuilders.begin(), itEnd = m_typeToBuilders.end();
       it != itEnd;
       ++it) {
      for ( std::vector<std::string>::const_iterator builderName = it->second.begin();
            builderName != it->second.end(); ++builderName )
      {
         if(builderName->substr(0,kSimple.size()) == kSimple) {
            returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>( new FWSimpleRepresentationChecker(
                                                                               builderName->substr(kSimple.size(),
                                                                                                   builderName->find_first_of('@')-kSimple.size()),
                                                                               it->first)));
         } else {

            returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>( new FWEDProductRepresentationChecker(
                                                                               builderName->substr(0,builderName->find_first_of('@')),
                                                                               it->first)));
         }
      }
   }
   return returnValue;
}

