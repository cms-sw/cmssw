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
// $Id: FWDetailViewBase.cc,v 1.11 2009/10/08 19:28:11 amraktad Exp $
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
FWDetailViewBase::makePackViewer(TEveWindowSlot *&slot, TGVerticalFrame *&guiFrame, TEveViewer *&eveViewer, TEveScene *&scene)
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

   // viewer GL
   TGLEmbeddedViewer *egl = new TGLEmbeddedViewer(pack, 0, 0);
   eveViewer= new TEveViewer("DetailViewViewer");
   eveViewer->SetGLViewer(egl, egl->GetFrame());
   gEve->GetViewers()->AddElement(eveViewer);
   pack->AddFrameWithWeight(egl->GetFrame(),0, 5);
   scene = gEve->SpawnNewScene("Detailed view");
   eveViewer->AddScene(scene);

   pack->MapSubwindows();
   pack->Layout();
   pack->MapWindow();

   return wp;
}
