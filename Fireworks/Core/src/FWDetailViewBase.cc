// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jan  9 13:35:56 EST 2009
// $Id: FWDetailViewBase.cc,v 1.18 2009/12/04 17:59:05 amraktad Exp $
//

// system include files
#include "TGPack.h"
#include "TCanvas.h"
#include "TBox.h"
#include "TEllipse.h"

#include "TRootEmbeddedCanvas.h"
#include "TGLEmbeddedViewer.h"
#include "TGLViewer.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveScene.h"
#include "RVersion.h"

// user include files
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"

FWDetailViewBase::FWDetailViewBase(const std::type_info& iInfo) :
   m_eveWindow(0),
   m_helper(iInfo)
{
}

FWDetailViewBase::~FWDetailViewBase()
{
   // Nothing to do here: DestroyWindowAndSlot clears gui components
   // detail view scene are set to auto destruct
}

void
FWDetailViewBase::build (const FWModelId &iID, TEveWindowSlot* slot)
{
   m_helper.itemChanged(iID.item());
   build(iID, m_helper.offsetObject(iID.item()->modelData(iID.index())), slot);
}

TEveWindow*
FWDetailViewBase::makePackCanvas(TEveWindowSlot *&slot, TCanvas *&infoCanvas, TCanvas *&viewCanvas)
{  TEveWindowPack* wp = slot->MakePack();
   wp->SetHorizontal();
   wp->SetShowTitleBar(kFALSE);
   
   // left canvas
   slot = wp->NewSlotWithWeight(1);
   TRootEmbeddedCanvas*  ecInfo = new TRootEmbeddedCanvas();
   TEveWindowFrame* wf = slot->MakeFrame(ecInfo);
   wf->GetEveFrame()->SetShowTitleBar(kFALSE);
   infoCanvas = ecInfo->GetCanvas();
   
   // view canvas
   slot = wp->NewSlotWithWeight(3);
   TRootEmbeddedCanvas*  ecV = new TRootEmbeddedCanvas();
   wf = slot->MakeFrame(ecV);
   wf->GetEveFrame()->SetShowTitleBar(kFALSE);
   viewCanvas = ecV->GetCanvas();
   

    return wp;
}

TEveWindow*
FWDetailViewBase::makePackViewer(TEveWindowSlot *&slot, TCanvas *&canvas, TEveViewer *&eveViewer, TEveScene *&scene)
{
   TEveWindowPack* wp = slot->MakePack();
   wp->SetHorizontal();
   wp->SetShowTitleBar(kFALSE);

   // left canvas
   slot = wp->NewSlotWithWeight(1);
   TRootEmbeddedCanvas*  ec = new TRootEmbeddedCanvas();
   TEveWindowFrame* wf = slot->MakeFrame(ec);
   wf->GetEveFrame()->SetShowTitleBar(kFALSE);
   canvas = ec->GetCanvas();
   canvas->SetHighLightColor(-1);

   // viewer GL
   slot = wp->NewSlotWithWeight(3);
   eveViewer = new TEveViewer("Detail view");
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,25,4)
   eveViewer->SpawnGLEmbeddedViewer(0);
#else
   eveViewer->SpawnGLEmbeddedViewer();
#endif
   gEve->GetViewers()->AddElement(eveViewer);
   slot->ReplaceWindow(eveViewer);
   slot->SetShowTitleBar(kFALSE);

   // scene
   scene = gEve->SpawnNewScene("Detailed view");
   eveViewer->AddScene(scene);

   return wp;
}

TEveWindow*
FWDetailViewBase::makePackViewerGui(TEveWindowSlot *&slot,  TCanvas *&canvas, TGVerticalFrame*& guiFrame, TEveViewer *&eveViewer, TEveScene *&scene)
{
   TEveWindowPack* wp = slot->MakePack();
   wp->SetHorizontal();
   wp->SetShowTitleBar(kFALSE);

   // canvas & widgets
   slot = wp->NewSlotWithWeight(1);
   TEveWindowFrame* wf = slot->MakeFrame();
   wf->SetShowTitleBar(kFALSE);
   TGCompositeFrame* eveFrame = wf->GetGUICompositeFrame();
   guiFrame = new TGVerticalFrame(eveFrame, 10, 10, kSunkenFrame|kDoubleBorder);
   eveFrame->AddFrame(guiFrame, new TGLayoutHints(kLHintsNormal| kLHintsExpandX));

   
   TGCompositeFrame* cf = new TGCompositeFrame(eveFrame);
   eveFrame->AddFrame(cf, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   cf->SetCleanup(kLocalCleanup);

   TRootEmbeddedCanvas* ec = new TRootEmbeddedCanvas("Embeddedcanvas", cf, 100, 100, 0);
   cf->AddFrame(ec, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   canvas = ec->GetCanvas();
   canvas->SetHighLightColor(-1);
   cf->Layout();
   
   eveFrame->MapSubwindows();
   eveFrame->Layout();
   
   // viewer GL 
   slot = wp->NewSlotWithWeight(3);
   eveViewer = new TEveViewer("Detail view");
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,25,4)
   eveViewer->SpawnGLEmbeddedViewer(0);
#else
   eveViewer->SpawnGLEmbeddedViewer();
#endif
   gEve->GetViewers()->AddElement(eveViewer);
   slot->ReplaceWindow(eveViewer);
   slot->SetShowTitleBar(kFALSE);
   scene = gEve->SpawnNewScene("Detailed view");
   eveViewer->AddScene(scene);
      
   return wp;
}

void
FWDetailViewBase::drawCanvasDot(Float_t x, Float_t y, Float_t r, Color_t fillColor)
{ 
   // utility function to draw outline cricle

   Float_t ratio = 0.5;
   // fill
   TEllipse *b2 = new TEllipse(x, y, r, r*ratio);
   b2->SetFillStyle(1001);
   b2->SetFillColor(fillColor);
   b2->Draw();

   // outline
   TEllipse *b1 = new TEllipse(x, y, r, r*ratio);
   b1->SetFillStyle(0);
   b1->SetLineWidth(2);
   b1->Draw();
}

void
FWDetailViewBase::drawCanvasBox( Double_t *pos, Color_t fillCol, Int_t fillType, bool bg)
{ 
   // utility function to draw outline box

   // background
   if (bg)
   {
      TBox *b1 = new TBox(pos[0], pos[1], pos[2], pos[3]);
      b1->SetFillColor(fillCol);
      b1->Draw();
   }

   // fill  (top layer)
   TBox *b2 = new TBox(pos[0], pos[1], pos[2], pos[3]);
   b2->SetFillStyle(fillType);
   b2->SetFillColor(kBlack);
   b2->Draw();

   //outline
   TBox *b3 = new TBox(pos[0], pos[1], pos[2], pos[3]);
   b3->SetFillStyle(0);
   b3->SetLineWidth(2);
   b3->Draw();
}


