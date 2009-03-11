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
// $Id: FW3DLegoView.cc,v 1.6 2009/01/23 21:35:42 amraktad Exp $
//

// system include files
#include <iostream>

#include "TRootEmbeddedCanvas.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TH2F.h"
#include "TView.h"
#include "TEveWindow.h"

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
FW3DLegoView::FW3DLegoView(TEveWindowSlot* iParent)
{
   TRootEmbeddedCanvas* eCanvas = new TRootEmbeddedCanvas("legoCanvas", 0);
   TEveWindowFrame *wf = iParent->MakeFrame(eCanvas);
   wf->SetElementName("FW3DLegoView");

   m_frame = eCanvas;
   m_legoCanvas = eCanvas->GetCanvas();

   m_legoCanvas->SetFillColor(Color_t(kBlack));


   // one way of connecting event processing function to a canvas
   // m_legoCanvas->AddExec("ex", "FW3DLegoView::DynamicCoordinates()");
}

void FW3DLegoView::connect( const char* receiver_class, void* receiver, const char* slot )
{
   // use Qt messaging mechanism
   m_legoCanvas->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
                         receiver_class, receiver, slot );
   // "FW3DLegoView", this,
   // "exec3event(Int_t,Int_t,Int_t,TObject*)");
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

void
FW3DLegoView::saveImageTo(const std::string& iName) const
{
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

