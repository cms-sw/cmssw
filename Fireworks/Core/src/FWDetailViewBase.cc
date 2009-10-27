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
// $Id: FWDetailViewBase.cc,v 1.12 2009/10/12 18:02:45 amraktad Exp $
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
FWDetailViewBase::makePackCanvas(TEveWindowSlot *&slot, TGVerticalFrame *&guiFrame, TCanvas *&viewCanvas)
{
   TEveWindowPack* wp = slot->MakePack();
   wp->SetShowTitleBar(kFALSE);
   TGPack* pack = wp->GetPack();
   pack->SetVertical(kFALSE);
   pack->SetUseSplitters(kFALSE);

   // gui frame 
   guiFrame = new TGVerticalFrame(pack, 10, 10, kSunkenFrame|kDoubleBorder);
   guiFrame->SetCleanup(kLocalCleanup);
   pack->AddFrameWithWeight(guiFrame, new TGLayoutHints(kLHintsNormal),2);

   // 2D canvas
   TRootEmbeddedCanvas*  ec   = new TRootEmbeddedCanvas("Embeddedcanvas", pack);
   pack->AddFrameWithWeight(ec, 0, 5);
   viewCanvas = ec->GetCanvas();
   viewCanvas->SetHighLightColor(-1);
   
   pack->MapSubwindows();
   pack->Layout();
   pack->MapWindow();

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
   slot->SetShowTitleBar(kFALSE);
   TRootEmbeddedCanvas*  ec = new TRootEmbeddedCanvas();
   slot->MakeFrame(ec);
   canvas = ec->GetCanvas();

   // viewer GL
   slot = wp->NewSlotWithWeight(3);
   slot->SetShowTitleBar(kFALSE);
   eveViewer = new TEveViewer("Detail view");
   eveViewer->SpawnGLEmbeddedViewer();   gEve->GetViewers()->AddElement(eveViewer);
   slot->ReplaceWindow(eveViewer);

   // scene
   scene = gEve->SpawnNewScene("Detailed view");
   eveViewer->AddScene(scene);

   return wp;
}
