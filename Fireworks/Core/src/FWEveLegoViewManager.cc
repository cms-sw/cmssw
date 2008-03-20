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
// $Id: FWEveLegoViewManager.cc,v 1.1.2.4 2008/03/18 01:40:04 dmytro Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include <algorithm>
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

// user include files
#include "Fireworks/Core/interface/FWEveLegoViewManager.h"
#include "Fireworks/Core/interface/FWEveLegoView.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"

#include "TEveSelection.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

//
// constants, enums and typedefs
//
const char* const FWEveLegoViewManager::m_builderPrefixes[] = {
   "Proxy3DLegoBuilder",
   "ProxyEveLegoBuilder",
};

//
// static data member definitions
//

//
// constructors and destructor
//
FWEveLegoViewManager::FWEveLegoViewManager(FWGUIManager* iGUIMgr):
FWViewManagerBase(m_builderPrefixes,
		  m_builderPrefixes+sizeof(m_builderPrefixes)/sizeof(const char*)),
  m_elements(0),
  m_data(0),
  m_legoRebinFactor(1),
  m_itemChanged(false),
  m_eveSelection(0),
  m_selectionManager(0)
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FWEveLegoViewManager::buildView,
                 this, _1);
   iGUIMgr->registerViewBuilder(FWEveLegoView::staticTypeName(), f);
   
   m_eveSelection=gEve->GetSelection();
   m_eveSelection->SetPickToSelect(TEveSelection::kPS_Projectable);
   m_eveSelection->Connect("SelectionAdded(TEveElement*)","FWEveLegoViewManager",this,"selectionAdded(TEveElement*)");
   m_eveSelection->Connect("SelectionRemoved(TEveElement*)","FWEveLegoViewManager",this,"selectionRemoved(TEveElement*)");
   m_eveSelection->Connect("SelectionCleared()","FWEveLegoViewManager",this,"selectionCleared()");
}

FWEveLegoViewManager::~FWEveLegoViewManager()
{
}

//
// member functions
//
FWViewBase* 
FWEveLegoViewManager::buildView(TGFrame* iParent)
{
   if ( ! m_elements ) m_elements = new TEveElementList("Lego");
   
   if(0==m_data) {
      m_data = new TEveCaloDataHist();
      // m_data->SetMaximum(100);
   }
   boost::shared_ptr<FWEveLegoView> view( new FWEveLegoView(iParent, m_elements) );
   m_views.push_back(view);
   return view.get();

}


void 
FWEveLegoViewManager::newEventAvailable()
{
  
   if(0==m_data || 0==m_views.size()) return;
   
   // m_data = new TEveCaloDataHist(); // it's a smart object, so it will clean up
   
   //   for ( std::vector<FWEveLegoModelProxy>::iterator proxy =  m_modelProxies.begin();
   //	 proxy != m_modelProxies.end(); ++proxy ) {
   for ( unsigned int i = 0; i < m_modelProxies.size(); ++i ) {
      if ( m_modelProxies[i].ignore ) continue;
      FWEveLegoModelProxy* proxy = & (m_modelProxies[i]);
      if ( proxy->product == 0) // first time
	{
	   TObject* product(0);
	   proxy->builder->build( &product );
	   if ( ! product) {
	      printf("WARNING: proxy builder failed to initialize product for FWEveLegoViewManager. Ignored\n");
	      proxy->ignore = true;
	      continue;
	   } 
	   TH2F* hist = dynamic_cast<TH2F*>(product);
	   if ( hist ) {
	      hist->Rebin2D(); // FIX ME
	      m_data->AddHistogram(hist);
	      // set color
	      TColor *color = gROOT->GetColor( hist->GetFillColor() );
	      new TColor( 1000+m_data->GetNSlices()-1, color->GetRed(), color->GetGreen(), color->GetBlue() );
	      // HACK to share a single histogram for each proxy
	      // TEveLegoDataHist creates an internal copy of the histogram, 
	      // let's use it for the product as well.
	      proxy->product = const_cast<TH2F*>(m_data->GetHistogram(m_data->GetNSlices()-1) );
	      continue;
	   }
	   TEveElementList* element = dynamic_cast<TEveElementList*>(product);
	   if ( element ) {
	      m_elements->AddElement( element );
	      proxy->product = element;
	      continue;
	   }
	   printf("WARNING: unknown product for FWEveLegoViewManager. Proxy is ignored\n");
	   proxy->ignore = true;
	} else {
	   proxy->builder->build( &(proxy->product) );
	}
   }

   std::for_each(m_views.begin(), m_views.end(),
                 boost::bind(&FWEveLegoView::draw,_1, m_data) );
}

void 
FWEveLegoViewManager::makeProxyBuilderFor(const FWEventItem* iItem)
{
   if(0==m_selectionManager) {
      //std::cout <<"got selection manager"<<std::endl;
      m_selectionManager = iItem->selectionManager();
   }
  TypeToBuilders::iterator itFind = m_typeToBuilders.find(iItem->name());
  if(itFind != m_typeToBuilders.end()) {
     for ( std::vector<std::string>::const_iterator builderName = itFind->second.begin();
	   builderName != itFind->second.end(); ++builderName )
       {
	  FW3DLegoDataProxyBuilder* builder = 
	    reinterpret_cast<FW3DLegoDataProxyBuilder*>(
							createInstanceOf(TClass::GetClass(typeid(FW3DLegoDataProxyBuilder)),
									 builderName->c_str())
							);
	  if(0!=builder) {
	     boost::shared_ptr<FW3DLegoDataProxyBuilder> pB( builder );
	     builder->setItem(iItem);
	     m_modelProxies.push_back(FWEveLegoModelProxy(pB) );
	  }
       }
  }
   iItem->itemChanged_.connect(boost::bind(&FWEveLegoViewManager::itemChanged,this,_1));
}

void 
FWEveLegoViewManager::newItem(const FWEventItem* iItem)
{
   makeProxyBuilderFor(iItem);
}

void 
FWEveLegoViewManager::registerProxyBuilder(const std::string& iType,
					   const std::string& iBuilder,
					   const FWEventItem* iItem)
{
   m_typeToBuilders[iType].push_back(iBuilder);
   if(0!=iItem) {
      makeProxyBuilderFor(iItem);
   }
}

void 
FWEveLegoViewManager::itemChanged(const FWEventItem*) {
   m_itemChanged=true;
}
void 
FWEveLegoViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}
void 
FWEveLegoViewManager::modelChangesDone()
{
   if ( m_itemChanged ) 
     newEventAvailable();
   else {
      std::for_each(m_views.begin(), m_views.end(),
		    boost::bind(&FWEveLegoView::draw,_1, m_data) );
   }
   
   m_itemChanged = false;
   gEve->EnableRedraw();
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
FWEveLegoViewManager::selectionCleared()
{
   if(0!= m_selectionManager) {
      m_selectionManager->clearSelection();
   }   
}

