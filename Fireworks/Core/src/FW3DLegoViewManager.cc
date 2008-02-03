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
// $Id: FW3DLegoViewManager.cc,v 1.6 2008/02/03 02:43:55 dmytro Exp $
//

// system include files
#include <iostream>
#include "THStack.h"
#include "TCanvas.h"
#include "TVirtualHistPainter.h"
#include "TH2F.h"
#include "TView.h"
#include "TList.h"
#include "TEveManager.h"
#include "TClass.h"
#include "TColor.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoViewManager.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DLegoViewManager::FW3DLegoViewManager():
  FWViewManagerBase("Proxy3DLegoBuilder"),
  m_legoCanvas(0),
  m_legoRebinFactor(1)
{
  m_legoCanvas = gEve->AddCanvasTab("legoCanvas");
   
   m_legoCanvas->SetFillColor(Color_t(kBlack));
     
  // one way of connecting event processing function to a canvas
  m_legoCanvas->AddExec("ex", "FW3DLegoViewManager::DynamicCoordinates()");
  
  // use Qt messaging mechanism
  m_legoCanvas->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
			"FW3DLegoViewManager", this,
			"exec3event(Int_t,Int_t,Int_t,TObject*)");

  m_stack = new THStack("LegoStack", "Calo tower lego plot");
  m_stack->SetMaximum(100);

  m_background = new TH2F("bkgLego","Background distribution",
			  82, fw3dlego::xbins, 72/m_legoRebinFactor, -3.1416, 3.1416);
   m_background->SetFillColor( Color_t(TColor::GetColor("#151515")) );
  m_background->Rebin2D();
  m_stack->Add(m_background);

  m_legoCanvas->cd();
  // m_stack->GetHistogram()->GetXaxis()->SetRangeUser(-1.74,1.74); // zoomed in default view
  
  m_stack->Draw("lego1 fb bb");
  m_legoCanvas->Modified();
  m_legoCanvas->Update();

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
void 
FW3DLegoViewManager::newEventAvailable()
{
  
   for ( std::vector<FW3DLegoModelProxy>::iterator proxy = 
	   m_modelProxies.begin();
	proxy != m_modelProxies.end(); ++proxy ) {
    bool firstTime = (proxy->product == 0);
    proxy->builder->build( &(proxy->product) );
    if(firstTime && 0!= proxy->product) {
       proxy->product->Rebin2D();
       m_stack->Add(proxy->product);
    }
  }

  m_legoCanvas->cd();
  
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
   
  m_stack->Draw("lego1 fb bb");
  m_legoCanvas->Modified();
  m_legoCanvas->Update();
}

void 
FW3DLegoViewManager::newItem(const FWEventItem* iItem)
{
  TypeToBuilder::iterator itFind = m_typeToBuilder.find(iItem->name());
  if(itFind != m_typeToBuilder.end()) {
    FW3DLegoDataProxyBuilder* builder = reinterpret_cast<
      FW3DLegoDataProxyBuilder*>( 
        createInstanceOf(TClass::GetClass(typeid(FW3DLegoDataProxyBuilder)),
			 itFind->second.c_str())
	);
    if(0!=builder) {
      boost::shared_ptr<FW3DLegoDataProxyBuilder> pB( builder );
      builder->setItem(iItem);
      m_modelProxies.push_back(FW3DLegoModelProxy(pB) );
    }
  }
}

void 
FW3DLegoViewManager::registerProxyBuilder(const std::string& iType,
					  const std::string& iBuilder)
{
  m_typeToBuilder[iType]=iBuilder;
}


void FW3DLegoViewManager::DynamicCoordinates()
{
   // Buttons.h
   int event = gPad->GetEvent();
   if (event != kButton1Down) return;
   std::cout << "Event: " << event << std::endl;
   gPad->GetCanvas()->FeedbackMode(kTRUE);
   
   int px = gPad->GetEventX();
   int py = gPad->GetEventY();
   std::cout << px << "   " << py << std::endl;
}
 
void FW3DLegoViewManager::exec3event(int event, int x, int y, TObject *selected)
{
   // Two modes of tower selection is supported:
   // - selection based on the base of a tower (point with z=0)
   // - project view of a tower (experimental)
   bool projectedMode = true;
   TCanvas *c = (TCanvas *) gTQSender;
   if (event == kButton2Down) {
      printf("Canvas %s: event=%d, x=%d, y=%d, selected=%s\n", c->GetName(),
	     event, x, y, selected->IsA()->GetName());
      THStack* stack = dynamic_cast<THStack*>(gPad->GetPrimitive("LegoStack"));
      if ( ! stack) return;
      TH2F* ecal = dynamic_cast<TH2F*>(stack->GetHists()->FindObject("ecalLego"));
      TH2F* hcal = dynamic_cast<TH2F*>(stack->GetHists()->FindObject("hcalLego"));
      // TH2F* h = dynamic_cast<TH2F*>(selected);
      if ( ecal && hcal ){
	 double zMax = 0.001;
	 if ( projectedMode ) zMax = stack->GetMaximum();
	 int selectedXbin(0), selectedYbin(0);
	 double selectedX(0), selectedY(0), selectedZ(0), selectedECAL(0), selectedHCAL(0);
	 
	 // scan non-zero z 
	 int oldx(0), oldy(0);
	 for ( double z = 0; z<zMax; z+=1) {
	    Double_t wcX,wcY;
	    pixel2wc(x,y,wcX,wcY,z);
	    int xbin = stack->GetXaxis()->FindFixBin(wcX);
	    int ybin = stack->GetYaxis()->FindFixBin(wcY);
	    if (oldx == xbin && oldy == ybin) continue;
	    oldx = xbin; 
	    oldy = ybin;
	    if ( xbin > stack->GetXaxis()->GetNbins() || ybin > stack->GetYaxis()->GetNbins() ) continue;
	    double content = ecal->GetBinContent(xbin,ybin)+hcal->GetBinContent(xbin,ybin);
	    if ( z <= content ) {
	       selectedXbin = xbin;
	       selectedYbin = ybin;
	       selectedX = wcX;
	       selectedY = wcY;
	       selectedZ = z;
	       selectedECAL = ecal->GetBinContent(xbin,ybin);
	       selectedHCAL = hcal->GetBinContent(xbin,ybin);
	    }
	 }
	 if ( selectedXbin > 0 && selectedYbin>0 )
	   std::cout << "x=" << selectedX << ", y=" << selectedY << ", z=" << selectedZ << 
	   ", xbin=" << selectedXbin << ", ybin=" << selectedYbin << ", Et (total, em, had): " <<  
	   selectedECAL+selectedHCAL << ", " << selectedECAL << ", " << selectedHCAL << std::endl;
      }
      /*
	TObject* obj = gPad->GetPrimitive("overLego");
      if ( ! obj ) return;
      if ( TH2F* h = dynamic_cast<TH2F*>(obj) )	{
	 h->SetBinContent(
      */
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
