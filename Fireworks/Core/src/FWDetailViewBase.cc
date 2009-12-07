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
// $Id: FWDetailViewBase.cc,v 1.14 2009/10/28 10:35:18 amraktad Exp $
//

// system include files
#include "TGPack.h"
#include "TCanvas.h"
#include "TRootEmbeddedCanvas.h"
#include "TGLEmbeddedViewer.h"
#include "TGLViewer.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveScene.h"

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
   eveViewer->SpawnGLEmbeddedViewer();
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
   eveFrame->AddFrame(guiFrame, new TGLayoutHints(kLHintsNormal));
   
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
   eveViewer->SpawnGLEmbeddedViewer();
   gEve->GetViewers()->AddElement(eveViewer);
   slot->ReplaceWindow(eveViewer);
   slot->SetShowTitleBar(kFALSE);
   scene = gEve->SpawnNewScene("Detailed view");
   eveViewer->AddScene(scene);
      
   return wp;
}