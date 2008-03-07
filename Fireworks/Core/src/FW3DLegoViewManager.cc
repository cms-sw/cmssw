// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoViewManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: FW3DLegoViewManager.cc,v 1.14 2008/03/06 10:17:17 dmytro Exp $
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

// user include files
#include "Fireworks/Core/interface/FW3DLegoViewManager.h"
#include "Fireworks/Core/interface/FW3DLegoView.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DLegoViewManager::FW3DLegoViewManager(FWGUIManager* iGUIMgr):
  FWViewManagerBase("Proxy3DLegoBuilder"),
  m_stack(0),
  m_legoRebinFactor(1)
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FW3DLegoViewManager::buildView,
                 this, _1);
   iGUIMgr->registerViewBuilder(FW3DLegoView::staticTypeName(), f);
   
}

// FW3DLegoViewManager::FW3DLegoViewManager(const FW3DLegoViewManager& rhs)
// {
//    // do actual copying here;
// }

FW3DLegoViewManager::~FW3DLegoViewManager()
{
}

//
// assignment operators
//
// const FW3DLegoViewManager& FW3DLegoViewManager::operator=(const FW3DLegoViewManager& rhs)
// {
//   //An exception safe implementation is
//   FW3DLegoViewManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
FWViewBase* 
FW3DLegoViewManager::buildView(TGFrame* iParent)
{
   if(0==m_stack) {
      m_stack = new THStack("LegoStack", "Calo tower lego plot");
      m_stack->SetMaximum(100);
      
      m_background = new TH2F("bkgLego","Background distribution",
                              82, fw3dlego::xbins, 72/m_legoRebinFactor, -3.1416, 3.1416);
      m_background->SetFillColor( Color_t(TColor::GetColor("#151515")) );
      
      m_highlight  = new TH2F("highLego","Highlight distribution",
                              82, fw3dlego::xbins, 72/m_legoRebinFactor, -3.1416, 3.1416);
      m_highlight->SetFillColor( kWhite );
      
      m_highlight_map  = new TH2C("highLego","Highlight distribution",
                              82, fw3dlego::xbins, 72/m_legoRebinFactor, -3.1416, 3.1416);
      m_background->Rebin2D();
      m_highlight->Rebin2D();
      m_highlight_map->Rebin2D();
      m_stack->Add(m_background);
      m_stack->Add(m_highlight);
   }
   boost::shared_ptr<FW3DLegoView> view( new FW3DLegoView(iParent) );
   m_views.push_back(view);
   view->draw(m_stack);
   return view.get();

}


void 
FW3DLegoViewManager::newEventAvailable()
{
  
   if(0==m_stack || 0==m_views.size()) return;
   
   m_highlight_map->Reset();
   m_highlight->Reset();
   for ( std::vector<FW3DLegoModelProxy>::iterator proxy =  m_modelProxies.begin();
	 proxy != m_modelProxies.end(); ++proxy ) {
      bool firstTime = (proxy->product == 0);
      proxy->builder->build( &(proxy->product) );
      if ( firstTime && 0!= proxy->product ){
	 proxy->product->Rebin2D();
	 if ( dynamic_cast<TH2F*>(proxy->product) ) m_stack->Add(proxy->product);
      }
      
      if ( proxy->product && dynamic_cast<TH2C*>(proxy->product) )
	m_highlight_map->Add(proxy->product);
  }

   // apply selection by moving data out of proxy products to m_highlight
   for ( int ix = 1; ix <= m_highlight_map->GetNbinsX(); ++ix ) {
      for ( int iy = 1; iy <= m_highlight_map->GetNbinsY(); ++iy ) {
	 if ( m_highlight_map->GetBinContent(ix,iy) < 1 ) continue;
	 for ( std::vector<FW3DLegoModelProxy>::iterator proxy =  m_modelProxies.begin();
	       proxy != m_modelProxies.end(); ++proxy ) {
	    if ( ! proxy->product ) continue;
	    m_highlight->SetBinContent(ix, iy, 
				       m_highlight->GetBinContent(ix,iy) + proxy->product->GetBinContent(ix,iy) 
				       );
	    proxy->product->SetBinContent(ix,iy,0);
	 }
      }
   }
   
  std::cout << "stack: " << m_stack << std::endl; 
   
  m_stack->GetHistogram()->GetXaxis()->SetTitle("#eta");
  m_stack->GetHistogram()->GetXaxis()->SetTitleColor(Color_t(kYellow));
  m_stack->GetHistogram()->GetYaxis()->SetTitle("#phi");
  m_stack->GetHistogram()->GetYaxis()->SetTitleColor(Color_t(kYellow));
  
  m_stack->GetHistogram()->GetXaxis()->SetLabelSize(0.03);
  m_stack->GetHistogram()->GetXaxis()->SetLabelColor(Color_t(kYellow));
  m_stack->GetHistogram()->GetXaxis()->SetAxisColor(Color_t(kYellow));
  m_stack->GetHistogram()->GetXaxis()->SetTickLength(0.02);
  m_stack->GetHistogram()->GetXaxis()->SetTitleOffset(1.2);
   
  m_stack->GetHistogram()->GetYaxis()->SetLabelSize(0.03);
  m_stack->GetHistogram()->GetYaxis()->SetLabelColor(Color_t(kYellow));
  m_stack->GetHistogram()->GetYaxis()->SetAxisColor(Color_t(kYellow));
  m_stack->GetHistogram()->GetYaxis()->SetTickLength(0.02);
  m_stack->GetHistogram()->GetYaxis()->SetTitleOffset(1.2);
  
  m_stack->GetHistogram()->GetZaxis()->SetTitle("Et, [GeV]");
  m_stack->GetHistogram()->GetZaxis()->SetTitleColor(Color_t(kYellow));
  m_stack->GetHistogram()->GetZaxis()->SetLabelColor(Color_t(kYellow));
  m_stack->GetHistogram()->GetZaxis()->SetAxisColor(Color_t(kYellow));
  m_stack->GetHistogram()->GetZaxis()->SetLabelSize(0.03);
  m_stack->GetHistogram()->GetZaxis()->SetTickLength(0.02); 
   
  m_stack->GetHistogram()->SetBit(TH1::kNoTitle);
   
   std::for_each(m_views.begin(), m_views.end(),
                 boost::bind(&FW3DLegoView::draw,_1, m_stack) );
}

void 
FW3DLegoViewManager::newItem(const FWEventItem* iItem)
{
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
	     m_modelProxies.push_back(FW3DLegoModelProxy(pB) );
	  }
       }
  }
}

void 
FW3DLegoViewManager::registerProxyBuilder(const std::string& iType,
					  const std::string& iBuilder)
{
   m_typeToBuilders[iType].push_back(iBuilder);
}

void 
FW3DLegoViewManager::modelChangesComing()
{
}
void 
FW3DLegoViewManager::modelChangesDone()
{
   newEventAvailable();
}

//
// const member functions
//

//
// static member functions
//
