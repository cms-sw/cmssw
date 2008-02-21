// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoView
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:41 EST 2008
// $Id: FW3DLegoView.cc,v 1.1 2008/02/21 19:20:04 chrjones Exp $
//

// system include files
#include <iostream>

#include "TRootEmbeddedCanvas.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TH2F.h"
#include "TView.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoView.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DLegoView::FW3DLegoView(TGFrame* iParent)
{
   TRootEmbeddedCanvas* eCanvas = new TRootEmbeddedCanvas("legoCanvas", iParent);
   m_frame = eCanvas;
   m_legoCanvas = eCanvas->GetCanvas(); 
   
   m_legoCanvas->SetFillColor(Color_t(kBlack));
   
   // one way of connecting event processing function to a canvas
   m_legoCanvas->AddExec("ex", "FW3DLegoView::DynamicCoordinates()");
   
   // use Qt messaging mechanism
   m_legoCanvas->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
                         "FW3DLegoView", this,
                         "exec3event(Int_t,Int_t,Int_t,TObject*)");
}

// FW3DLegoView::FW3DLegoView(const FW3DLegoView& rhs)
// {
//    // do actual copying here;
// }

FW3DLegoView::~FW3DLegoView()
{
}

//
// assignment operators
//
// const FW3DLegoView& FW3DLegoView::operator=(const FW3DLegoView& rhs)
// {
//   //An exception safe implementation is
//   FW3DLegoView temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FW3DLegoView::draw(THStack* iStack)
{
   m_legoCanvas->cd();
   iStack->Draw("lego1 fb bb");
   m_legoCanvas->Modified();
   m_legoCanvas->Update();   
   
}


void FW3DLegoView::DynamicCoordinates()
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

void FW3DLegoView::exec3event(int event, int x, int y, TObject *selected)
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

void FW3DLegoView::pixel2wc(const Int_t pixelX, const Int_t pixelY, 
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

//
// const member functions
//
TGFrame* 
FW3DLegoView::frame() const
{
   return m_frame;
}

const std::string& 
FW3DLegoView::typeName() const
{
   return staticTypeName();
}

//
// static member functions
//
const std::string& 
FW3DLegoView::staticTypeName()
{
   static std::string s_name("3D Lego");
   return s_name;
}

