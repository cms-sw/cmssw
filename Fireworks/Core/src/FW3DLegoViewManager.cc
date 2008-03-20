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
// $Id$
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
const char* const FW3DLegoViewManager::m_builderPrefixes[] = {
   "Proxy3DLegoBuilder",
   "ProxyTH2LegoBuilder",
};

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DLegoViewManager::FW3DLegoViewManager(FWGUIManager* iGUIMgr):
  FWViewManagerBase(m_builderPrefixes,
		    m_builderPrefixes+sizeof(m_builderPrefixes)/sizeof(const char*)),
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
   view->connect("FW3DLegoViewManager", this, "exec3event(Int_t,Int_t,Int_t,TObject*)");
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
      if ( proxy->ignore ) continue;
      bool firstTime = (proxy->product == 0);
      proxy->builder->build( &(proxy->product) );
      
      if ( ! proxy->product ){
	 printf("WARNING: proxy builder failed to initialize product for FW3DLegoViewManager. Ignored\n");
	 proxy->ignore = true;
	 continue;
      }

      if ( firstTime ){
	 ((TH2*)proxy->product)->Rebin2D();
	 if (TH2F* hist = dynamic_cast<TH2F*>(proxy->product) ) m_stack->Add(hist);
      }
      if ( TH2C* hist = dynamic_cast<TH2C*>(proxy->product) ) m_highlight_map->Add(hist);
  }

   // apply selection by moving data out of proxy products to m_highlight
   for ( int ix = 1; ix <= m_highlight_map->GetNbinsX(); ++ix ) {
      for ( int iy = 1; iy <= m_highlight_map->GetNbinsY(); ++iy ) {
	 if ( m_highlight_map->GetBinContent(ix,iy) < 1 ) continue;
	 for ( std::vector<FW3DLegoModelProxy>::iterator proxy =  m_modelProxies.begin();
	       proxy != m_modelProxies.end(); ++proxy ) {
	    if ( TH2F* product = dynamic_cast<TH2F*>(proxy->product) ) {
	       m_highlight->SetBinContent(ix, iy, 
					  m_highlight->GetBinContent(ix,iy) + product->GetBinContent(ix,iy)
					  );
	       product->SetBinContent(ix,iy,0);
	    }
	 }
      }
   }
   
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
FW3DLegoViewManager::makeProxyBuilderFor(const FWEventItem* iItem)
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
FW3DLegoViewManager::newItem(const FWEventItem* iItem)
{
   makeProxyBuilderFor(iItem);
}

void 
FW3DLegoViewManager::registerProxyBuilder(const std::string& iType,
					  const std::string& iBuilder,
                                          const FWEventItem* iItem)
{
   m_typeToBuilders[iType].push_back(iBuilder);
   if(0!=iItem) {
      makeProxyBuilderFor(iItem);
   }
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

void FW3DLegoViewManager::exec3event(int event, int x, int y, TObject *selected)
{
   // Two modes of tower selection is supported:
   // - selection based on the base of a tower (point with z=0)
   // - project view of a tower (experimental)
   bool projectedMode = true;
   TCanvas *c = (TCanvas *) gTQSender;
   if (event == kButton1Down || event == kButton2Down) {
      printf("Canvas %s: event=%d, x=%d, y=%d, selected=%s\n", c->GetName(),
	     event, x, y, selected->IsA()->GetName());
      if ( ! m_stack) return;
      
      double zMax = 0.001;
      if ( projectedMode ) zMax = m_stack->GetMaximum();
      int selectedXbin(0), selectedYbin(0);
      double selectedX(0), selectedY(0), selectedZ(0), selectedValue(0);
	 
      // scan non-zero z 
      int oldx(0), oldy(0);
      for ( double z = 0; z<zMax; z+=1) {
	 Double_t wcX,wcY;
	 pixel2wc(x,y,wcX,wcY,z);
	 int xbin = m_stack->GetXaxis()->FindFixBin(wcX);
	 int ybin = m_stack->GetYaxis()->FindFixBin(wcY);
	 if (oldx == xbin && oldy == ybin) continue;
	 oldx = xbin; 
	 oldy = ybin;
	 if ( xbin > m_stack->GetXaxis()->GetNbins() || ybin > m_stack->GetYaxis()->GetNbins() ) continue;
	 double content = 0;
	 TListIter next(m_stack->GetHists());
	 while ( TH2* layer = dynamic_cast<TH2*>(next()) ) content += layer->GetBinContent(xbin,ybin);
	 if ( z <= content ) {
	    selectedXbin = xbin;
	    selectedYbin = ybin;
	    selectedX = wcX;
	    selectedY = wcY;
	    selectedZ = z;
	    selectedValue = content;
	 }
      }
      if ( selectedXbin > 0 && selectedYbin>0 )	{
	 std::cout << "x=" << selectedX << ", y=" << selectedY << ", z=" << selectedZ << 
	   ", xbin=" << selectedXbin << ", ybin=" << selectedYbin << ", Et: " <<  
	   selectedValue << std::endl;
	 for ( std::vector<FW3DLegoModelProxy>::iterator proxy =  m_modelProxies.begin();
	       proxy != m_modelProxies.end(); ++proxy )
	   proxy->builder->message(event, selectedXbin, selectedYbin);
      } else {
	 for ( std::vector<FW3DLegoModelProxy>::iterator proxy =  m_modelProxies.begin();
	       proxy != m_modelProxies.end(); ++proxy )
	   proxy->builder->message(0,0,0);
      }
   }
}

void FW3DLegoViewManager::pixel2wc(const Int_t pixelX, const Int_t pixelY, 
                                   Double_t& wcX, Double_t& wcY, const Double_t wcZ)
{
   // we need to make Pixel to NDC to WC transformation with the following constraint:
   // - Pixel only gives 2 coordinates, so we don't know z coordinate in NDC
   // - We know that in WC z has specific value (depending on what we want to use as 
   //   a selection point. In the case of the base of each bin, z(wc) = 0
   // we need to solve some simple linear equations to get what we need
   Double_t ndcX, ndcY;
   ((TPad *)gPad)->AbsPixeltoXY( pixelX, pixelY, ndcX, ndcY); // Pixel to NDC
   Double_t* m = gPad->GetView()->GetTback(); // NDC to WC matrix
   double part1 = wcZ-m[11]-m[8]*ndcX-m[9]*ndcY;
   wcX = m[3] + m[0]*ndcX + m[1]*ndcY + m[2]/m[10]*part1;
   wcY = m[7] + m[4]*ndcX + m[5]*ndcY + m[6]/m[10]*part1;
}

