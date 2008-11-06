// -*- C++ -*-
//
// Package:     Core
// Class  :     ElectronView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: ElectronView.cc,v 1.3 2008/09/29 18:00:23 amraktad Exp $
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
#include "TEveViewer.h"
#include "TEveProjectionManager.h"
#include "TEveScene.h"
#include "TGFrame.h"
#include "TGLViewer.h"
#include "TGLEmbeddedViewer.h"
#include "TGButton.h"
#include "TClass.h"
#include "TColor.h"

// user include files
#include "Fireworks/Core/interface/ElectronView.h"
#include "Fireworks/Core/interface/ElectronsProxySCBuilder.h"
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
ElectronView::ElectronView()
     : frame(0)
{

}

void ElectronView::close_wm ()
{
     printf("mmmm, flaming death!\n");
     frame = 0;
}

void ElectronView::close_button ()
{
     printf("mmmm, flaming death!\n");
     frame->CloseWindow();
     frame = 0;
}

void ElectronView::event ()
{
     printf("ElectronView::event()\n");
     // make a frame
     frame = new // TGTransientFrame(0, gEve->GetBrowser(), 400, 400);
	  TGMainFrame(0, 400, 420);
     // connect the close-window button to something useful
     frame->Connect("CloseWindow()", "ElectronView", this, "close_wm()");
     frame->SetCleanup(kDeepCleanup);
     TGLEmbeddedViewer* v = new TGLEmbeddedViewer(frame, 0, 0);
     TEveViewer* nv = new TEveViewer();
     nv->SetGLViewer(v);
     nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
     // nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
     nv->GetGLViewer()->SetStyle(TGLRnrCtx::kOutline);
     nv->GetGLViewer()->SetClearColor(kBlack);
     ns = gEve->SpawnNewScene("Electron");
     nv->AddScene(ns);
     frame->AddFrame(v->GetFrame(),
		     new TGLayoutHints(kLHintsTop | kLHintsExpandX
				       | kLHintsExpandY));
     TGTextButton* exit_butt = new TGTextButton(frame, "Eat flaming death");
     exit_butt->Resize(20, 20);
     exit_butt->Connect("Clicked()", "ElectronView", this, "close_button()");
     frame->AddFrame(exit_butt, new TGLayoutHints(kLHintsTop | kLHintsExpandX));
     frame->Layout();
     frame->SetWindowName("Ooogadooga-WINDOW");
     frame->SetIconName("Ooogadooga-ICON");
     frame->MapSubwindows();
     frame->MapWindow();

     TEveElementList *list = 0;
     ElectronsProxySCBuilder::the_electron_sc_proxy->build(&list);
     gEve->AddElement(list, ns);
}

// ElectronView::ElectronView(const ElectronView& rhs)
// {
//    // do actual copying here;
// }

ElectronView::~ElectronView()
{
}

//
// assignment operators
//
// const ElectronView& ElectronView::operator=(const ElectronView& rhs)
// {
//   //An exception safe implementation is
//   ElectronView temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
