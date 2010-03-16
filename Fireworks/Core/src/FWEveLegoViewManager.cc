// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveLegoViewManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: FWEveLegoViewManager.cc,v 1.40 2010/03/16 12:25:25 amraktad Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include <algorithm>
#include "Math/Math.h" //defined M_PI
#include "THStack.h"
#include "TCanvas.h"
#include "TVirtualHistPainter.h"
#include "TH2F.h"
#include "TView.h"
#include "TList.h"
#include "TEveManager.h"
#include "TClass.h"
#include "TColor.h"
#include "TRootEmbeddedCanvas.h"
#include "TEveCaloData.h"
#include "TEveElement.h"
#include "TROOT.h"
#include "TEveStraightLineSet.h"

#include "TEveTrans.h"

// user include files
#include "Fireworks/Core/interface/FWEveLegoViewManager.h"
#include "Fireworks/Core/interface/FWEveLegoView.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"

#include "TEveSelection.h"
#include "TEveCalo.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilderFactory.h"
#include "Fireworks/Core/interface/FWEDProductRepresentationChecker.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"

#include "Fireworks/Core/interface/fw3dlego_xbins.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEveLegoViewManager::FWEveLegoViewManager(FWGUIManager* iGUIMgr) :
   FWViewManagerBase(),
   m_elements( new TEveElementList("Lego")),
   m_data(0),
   m_lego(0),
   m_boundaries(0),
   m_eveSelection(0),
   m_selectionManager(0),
   m_modelsHaveBeenMadeAtLeastOnce(false)
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FWEveLegoViewManager::buildView,
                 this, _1);
   iGUIMgr->registerViewBuilder(FWEveLegoView::staticTypeName(), f);

   //create a list of the available ViewManager's
   std::set<std::string> builders;

   std::vector<edmplugin::PluginInfo> available = FW3DLegoDataProxyBuilderFactory::get()->available();
   std::transform(available.begin(),
                  available.end(),
                  std::inserter(builders,builders.begin()),
                  boost::bind(&edmplugin::PluginInfo::name_,_1));

   if(edmplugin::PluginManager::get()->categoryToInfos().end()!=edmplugin::PluginManager::get()->categoryToInfos().find(FW3DLegoDataProxyBuilderFactory::get()->category())) {
      available = edmplugin::PluginManager::get()->categoryToInfos().find(FW3DLegoDataProxyBuilderFactory::get()->category())->second;
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

FWEveLegoViewManager::~FWEveLegoViewManager()
{
   m_data->DecDenyDestroy();
}

//
// member functions
//
void
FWEveLegoViewManager::initData()
{
   if(0==m_data) {
      m_data = new TEveCaloDataHist();
      m_data->IncDenyDestroy();
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      TH2F* background = new TH2F("background","",
                                  82, fw3dlego::xbins, 72/1, -3.1416, 3.1416);
      TH1::AddDirectory(status);
      m_data->AddHistogram(background);
   }
}

FWViewBase*
FWEveLegoViewManager::buildView(TEveWindowSlot* iParent)
{
   initData();
   boost::shared_ptr<FWEveLegoView> view( new FWEveLegoView(iParent, m_elements.get()) );
   view->setBackgroundColor(colorManager().background());
   view->beingDestroyed_.connect(boost::bind(&FWEveLegoViewManager::beingDestroyed,this,_1));
   m_views.push_back(view);

   if(1 == m_views.size()) {
      for(std::vector<boost::shared_ptr<FW3DLegoDataProxyBuilder> >::iterator it
             =m_builders.begin(), itEnd = m_builders.end();
          it != itEnd;
          ++it) {
         (*it)->setHaveAWindow(true);
      }
   }
   view->beingDestroyed_.connect(boost::bind(&FWEveLegoViewManager::beingDestroyed,this,_1));

   //Do this ONLY if we've already added items to the scene else we get seg faults :(
   if(m_modelsHaveBeenMadeAtLeastOnce) {
      view->finishSetup();
   }
   return view.get();
}

void
FWEveLegoViewManager::beingDestroyed(const FWViewBase* iView)
{
   if(1 == m_views.size()) {
      for(std::vector<boost::shared_ptr<FW3DLegoDataProxyBuilder> >::iterator it
             =m_builders.begin(), itEnd = m_builders.end();
          it != itEnd;
          ++it) {
         (*it)->setHaveAWindow(false);
      }
   }
   for(std::vector<boost::shared_ptr<FWEveLegoView> >::iterator it=
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
FWEveLegoViewManager::makeProxyBuilderFor(const FWEventItem* iItem)
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
         FW3DLegoDataProxyBuilder* builder = FW3DLegoDataProxyBuilderFactory::get()->create(*builderName);
         if(0!=builder) {
            boost::shared_ptr<FW3DLegoDataProxyBuilder> pB( builder );
            builder->setItem(iItem);
            initData();
            if(!m_lego) {
               m_lego = new TEveCaloLego(m_data);
               m_lego->InitMainTrans();
               m_lego->RefMainTrans().SetScale(2*M_PI, 2*M_PI, M_PI);
               m_lego->Set2DMode(TEveCaloLego::kValSize);
	       m_lego->SetDrawNumberCellPixels(20);
               m_data->GetEtaBins()->SetTitleFont(120);
               m_data->GetEtaBins()->SetTitle("h");
               m_data->GetPhiBins()->SetTitleFont(120);
               m_data->GetPhiBins()->SetTitle("f");
               m_data->GetPhiBins()->SetLabelSize(0.02);
               m_data->GetEtaBins()->SetLabelSize(0.02);
               m_data->GetPhiBins()->SetTitleSize(0.03);
               m_data->GetEtaBins()->SetTitleSize(0.03);
               m_elements->AddElement(m_lego);

               // add calorimeter boundaries
               m_boundaries = new TEveStraightLineSet("boundaries");
               m_boundaries->SetPickable(kFALSE);
               m_boundaries->SetLineWidth(2);
               m_boundaries->SetLineStyle(7);
               m_boundaries->AddLine(-1.479,-3.1416,0.001,-1.479,3.1416,0.001);
               m_boundaries->AddLine(1.479,-3.1416,0.001,1.479,3.1416,0.001);
               m_boundaries->AddLine(-2.964,-3.1416,0.001,-2.964,3.1416,0.001);
               m_boundaries->AddLine(2.964,-3.1416,0.001,2.964,3.1416,0.001);
               m_lego->AddElement(m_boundaries);

               setGridColors();
            }
            builder->attach(m_elements.get(),m_data);
            m_builders.push_back(pB);
            if(m_views.size()) {
               pB->setHaveAWindow(true);
            }
         }
      }
   }
}

void
FWEveLegoViewManager::newItem(const FWEventItem* iItem)
{
   makeProxyBuilderFor(iItem);
}

void
FWEveLegoViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}

void
FWEveLegoViewManager::modelChangesDone()
{
   m_modelsHaveBeenMadeAtLeastOnce=true;
   std::for_each(m_views.begin(), m_views.end(),
                 boost::bind(&FWEveLegoView::finishSetup,_1) );
   gEve->EnableRedraw();
}

void
FWEveLegoViewManager::setGridColors()
{
   if(m_lego) {
      m_lego->SetFontColor(colorManager().geomColor(kFWLegoFontColorIndex));
      m_lego->SetGridColor(colorManager().geomColor(kFWLegoFrameColorIndex));
      m_boundaries->SetLineColor(colorManager().geomColor(kFWLegoBoundraryColorIndex));
   }
}

void
FWEveLegoViewManager::colorsChanged()
{
   setGridColors();
   std::for_each(m_views.begin(), m_views.end(),
                 boost::bind(&FWEveLegoView::setBackgroundColor,_1,colorManager().background()) );
}

void
FWEveLegoViewManager::selectionAdded(TEveElement* iElement)
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
FWEveLegoViewManager::selectionRemoved(TEveElement* iElement)
{
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
FWEveLegoViewManager::selectionCleared()
{
   if(0!= m_selectionManager) {
      m_selectionManager->clearSelection();
   }
}

FWTypeToRepresentations
FWEveLegoViewManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;
   const std::string kSimple("simple#");

   for(TypeToBuilders::const_iterator it = m_typeToBuilders.begin(), itEnd = m_typeToBuilders.end();
       it != itEnd;
       ++it) {
      for ( std::vector<std::string>::const_iterator builderName = it->second.begin();
            builderName != it->second.end(); ++builderName ) {
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

void
FWEveLegoViewManager::eventEnd()
{  
   for ( std::vector<boost::shared_ptr<FWEveLegoView> >::iterator view = m_views.begin(); view != m_views.end(); ++view )  
      (*view)->eventEnd();
}

