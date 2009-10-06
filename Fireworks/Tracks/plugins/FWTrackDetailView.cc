// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWTrackDetailView
// $Id: FWTrackDetailView.cc,v 1.18 2009/09/21 21:42:32 amraktad Exp $
//

#include "TEveLegoEventHandler.h"

// ROOT includes
#include "TLatex.h"
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TEveManager.h"
#include "TRootEmbeddedCanvas.h"

// CMSSW includes
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

// Fireworks includes
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/plugins/FWTrackDetailView.h"
#include "Fireworks/Tracks/interface/FWTrackResidualDetailView.h"
#include "Fireworks/Tracks/interface/FWTrackHitsDetailView.h"

//
// constructors and destructor
//
FWTrackDetailView::FWTrackDetailView()
{
}

FWTrackDetailView::~FWTrackDetailView()
{
}

//
// member functions
//
void FWTrackDetailView::build(const FWModelId &id, const reco::Track* iTrack, TEveWindowSlot* base)
{
   if(0==iTrack) return;
   TEveWindowPack* eveWindow = base->MakePack();
   TEveCompositeFrame* eveFrame = eveWindow->GetEveFrame();
   TGMainFrame* parent  = (TGMainFrame*)eveFrame->GetParent();
   parent->Resize(790, 450);
   eveFrame->Layout();
   eveWindow->SetShowTitleBar(kFALSE);
   eveWindow->SetHorizontal();
   FWDetailViewBase::setEveWindow(eveWindow);

   TEveWindowSlot* slot;
   ////////////////////////////////////////////////////////////////////////
   //                              Sub-view 1
   ///////////////////////////////////////////////////////////////////////
  
   // prepare window
   slot = eveWindow->NewSlotWithWeight(40);
   FWTrackResidualDetailView builder1;
   builder1.build(id,iTrack,slot);
 
   ////////////////////////////////////////////////////////////////////////
   //                              Sub-view 2
   ///////////////////////////////////////////////////////////////////////
   slot = eveWindow->NewSlotWithWeight(60);
   FWTrackHitsDetailView builder2;
   builder2.build(id,iTrack,slot);
}

REGISTER_FWDETAILVIEW(FWTrackDetailView);
